import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from datetime import datetime

# Ensure the AWQ .cu file is cleaned (no ../matmul.h) before running
this_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Load Extensions
w4a16_cuda_ext = load(
    name="w4a16_cuda_ext",
    sources=[os.path.join(this_dir, "w4a16_cuda.cu")],
    extra_cuda_cflags=["-O3", "-allow-unsupported-compiler"],
    verbose=True,
)

awq_cuda_ext = load(
    name="awq_cuda_ext",
    sources=[os.path.join(this_dir, "awq_kernel.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)

DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

# ==============================================================================
# 2. Kernel Wrappers
# ==============================================================================

def raw_cuda_w4a16(W, b, S, Z, group_size, activations):
    return w4a16_cuda_ext.forward(W.contiguous(), b.contiguous(), S.contiguous(), Z.contiguous(), activations.contiguous(), group_size)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_IF': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_IF': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_IF': 512}, num_warps=8),
    ],
    key=['OF', 'IF'],
)
@triton.jit
def _w4a16_gemv_kernel(
    W_ptr, b_ptr, S_ptr, Z_ptr, act_ptr, OUT_ptr,
    W_stride_0, W_stride_1, S_stride_0, S_stride_1,
    Z_stride_0, Z_stride_1, act_stride_0, OUT_stride_0,
    OF_packed, IF, group_size,
    BLOCK_SIZE_IF: tl.constexpr
):
    actual_of = tl.program_id(axis=0)
    pid_of_packed = actual_of // 2
    pid_half = actual_of % 2
    acc = 0.0
    for i in range(0, IF, BLOCK_SIZE_IF):
        offsets_if = i + tl.arange(0, BLOCK_SIZE_IF)
        mask_if = offsets_if < IF
        w_ptr = W_ptr + pid_of_packed * W_stride_0 + offsets_if * W_stride_1
        W = tl.load(w_ptr, mask=mask_if, other=0)
        g_id = offsets_if // group_size
        S = tl.load(S_ptr + actual_of * S_stride_0 + g_id * S_stride_1, mask=mask_if, other=1.0)
        Z = tl.load(Z_ptr + actual_of * Z_stride_0 + g_id * Z_stride_1, mask=mask_if, other=0.0)
        w_4bit = tl.where(pid_half == 0, W & 0xF, (W >> 4) & 0xF)
        w_fp16 = (w_4bit.to(tl.float32) - Z.to(tl.float32)) * S.to(tl.float32)
        act = tl.load(act_ptr + offsets_if * act_stride_0, mask=mask_if, other=0.0)
        acc += tl.sum(w_fp16 * act.to(tl.float32))
    bias = tl.load(b_ptr + actual_of)
    acc += bias.to(tl.float32)
    tl.store(OUT_ptr + actual_of * OUT_stride_0, acc.to(tl.float16))

def w4a16_gemv_triton(W, b, S, Z, group_size, activations):
    OF_packed, IF = W.shape
    OF_actual = OF_packed * 2
    OUT = torch.empty((OF_actual, 1), device=W.device, dtype=torch.float16)
    grid = lambda meta: (OF_actual,)
    _w4a16_gemv_kernel[grid](
        W, b, S, Z, activations, OUT,
        W.stride(0), W.stride(1), S.stride(0), S.stride(1),
        Z.stride(0), Z.stride(1), activations.stride(0), OUT.stride(0),
        OF_packed, IF, group_size
    )
    return OUT

# ==============================================================================
# 3. PyTorch Reference
# ==============================================================================

def dequantize_layer(W_q, S, Z, group_size):
    N, K = W_q.shape
    W_q_reshaped = W_q.view(N, K // group_size, group_size)
    W_deq = (W_q_reshaped.to(torch.float16) - Z.unsqueeze(-1)) * S.unsqueeze(-1)
    return W_deq.view(N, K)

def torch_w4a16(W_packed, b, S, Z, group_size, activations):
    w0, w1 = W_packed & 0x0F, (W_packed >> 4) & 0x0F
    W_unpacked = torch.stack([w0, w1], dim=1).view(-1, W_packed.shape[1])
    W_deq = dequantize_layer(W_unpacked, S, Z, group_size)
    out = torch.matmul(W_deq, activations) + b[:, None]
    return out

# ==============================================================================
# 4. Main Benchmarking Logic
# ==============================================================================

def plotting_and_benchmarking():
    OF_vals = [4096, 8192, 16384, 32768]
    IF = 8192
    B = 1
    group_size = 128
    PACK_FACTOR = 8
    records = []

    for OF in OF_vals:
        print(f"\n--- Benchmarking OF={OF} ---")
        
        # --- Data for Triton / Raw CUDA / Torch (Your Layout) ---
        W_packed = torch.randint(0, 255, (OF // 2, IF), device=DEVICE, dtype=torch.uint8).contiguous()
        b = torch.randn((OF,), device=DEVICE, dtype=torch.float16).contiguous()
        S = torch.ones((OF, IF // group_size), device=DEVICE, dtype=torch.float16).contiguous()
        Z = torch.randint(0, 15, (OF, IF // group_size), device=DEVICE, dtype=torch.float16).contiguous()
        act = torch.randn((IF, B), device=DEVICE, dtype=torch.float16).contiguous()
        W_ref_fp16 = torch.randn((OF, IF), device=DEVICE, dtype=torch.float16).contiguous()

        # --- Correctness check: raw CUDA vs Torch reference ---
        torch_ref = torch_w4a16(W_packed, b, S, Z, group_size, act)
        cuda_out = raw_cuda_w4a16(W_packed, b, S, Z, group_size, act)

        max_abs_err = (cuda_out - torch_ref).abs().max().item()
        mean_abs_err = (cuda_out - torch_ref).abs().mean().item()

        print(f"raw_cuda vs torch | max abs err = {max_abs_err:.6f}, mean abs err = {mean_abs_err:.6f}")

        assert torch.allclose(cuda_out, torch_ref, atol=1e-2, rtol=1e-2), (
            f"raw_cuda correctness check failed for OF={OF}: "
            f"max_abs_err={max_abs_err:.6f}, mean_abs_err={mean_abs_err:.6f}"
        )
        # --- Data for AWQ (Corrected Layout & Types) ---
        
        # 1. W_awq: Must be uint8 to pass PyTorch checks, but must allocate enough bytes 
        # to prevent the C++ uint32_t* cast from reading out of bounds.
        # (IF // 8 elements) * 4 bytes per element = IF // 2 bytes total.
        W_awq = torch.randint(0, 255, (OF, IF // 2), device=DEVICE, dtype=torch.uint8).contiguous()
        
        # 2. Z_awq: Compress into int32 first to do the bitwise logic safely
        Z_int = Z.to(torch.int32)
        Z_awq_int32 = torch.zeros((OF, (IF // group_size) // PACK_FACTOR), device=DEVICE, dtype=torch.int32)
        for i in range(PACK_FACTOR):
            Z_awq_int32 |= (Z_int[:, i::PACK_FACTOR] & 0xF) << (i * 4)
            
        # 3. Trick PyTorch: Cast the underlying memory back to a Byte tensor 
        # This turns an [OF, X] int32 tensor into an [OF, X * 4] uint8 tensor seamlessly!
        Z_awq = Z_awq_int32.view(torch.uint8).contiguous()

        S_awq = S.clone().contiguous()
        
        # Transpose activations to [B, IF] for AWQ
        act_awq = act.t().contiguous()
        # --- Benchmarking ---
        quantiles = [0.5, 0.05, 0.95]
        
        torch_ms = triton.testing.do_bench(lambda: torch.matmul(W_ref_fp16, act) + b[:, None], quantiles=quantiles)[0]
        tri_ms = triton.testing.do_bench(lambda: w4a16_gemv_triton(W_packed, b, S, Z, group_size, act), quantiles=quantiles)[0]
        cuda_ms = triton.testing.do_bench(lambda: raw_cuda_w4a16(W_packed, b, S, Z, group_size, act), quantiles=quantiles)[0]
        
        try:
            awq_ms = triton.testing.do_bench(lambda: awq_cuda_ext.forward(act_awq, W_awq, Z_awq, S_awq, IF, OF, group_size), quantiles=quantiles)[0]
        except Exception as e:
            print(f"AWQ Error: {e}")
            awq_ms = float('nan')

        flops = 2.0 * OF * IF * B
        for name, ms in [("torch", torch_ms), ("triton", tri_ms), ("raw_cuda", cuda_ms), ("awq", awq_ms)]:
            if not np.isnan(ms):
                records.append({"OF": OF, "provider": name, "ms": ms, "tflops": (flops * 1e-12) / (ms * 1e-3)})

    # --- Plotting ---
    df = pd.DataFrame(records)
    pivot_df = df.pivot(index='OF', columns='provider', values='tflops')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {"torch": "gray", "triton": "tab:blue", "raw_cuda": "tab:green", "awq": "tab:red"}
    
    for provider in df.provider.unique():
        sub = df[df.provider == provider]
        ax.plot(sub.OF, sub.tflops, marker="o", label=provider.upper(), color=colors.get(provider))

        if provider != "torch":
            for of_val in sub.OF:
                try:
                    speedup = pivot_df.loc[of_val, provider] / pivot_df.loc[of_val, "torch"]
                    ax.annotate(f"{speedup:.2f}x", xy=(of_val, pivot_df.loc[of_val, provider]), 
                                 xytext=(0, 10), textcoords="offset points", ha='center', fontweight='bold')
                except KeyError:
                    continue

    ax.set_title(f"W4A16 GEMV Performance Comparison (IF={IF}, B={B})")
    ax.set_ylabel("TFLOPS")
    ax.set_xlabel("Output Features")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(f"benchmark_full_{datetime.now().strftime('%H%M%S')}.png")
    plt.show()

if __name__ == "__main__":
    plotting_and_benchmarking()