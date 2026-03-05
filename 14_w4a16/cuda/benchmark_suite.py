import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from datetime import datetime

DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

# ==============================================================================
# 1. Load Custom CUDA Extension
# ==============================================================================
print("Loading custom CUDA extension...")
w4a16_cuda_ext = load(
    name='w4a16_cuda_ext', 
    sources=['w4a16_cuda.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math'], 
    verbose=True
)
print("CUDA Extension Loaded!")

def raw_cuda_w4a16(W, b, S, Z, group_size, activations):
    W = W.contiguous()
    b = b.contiguous()
    S = S.contiguous()
    Z = Z.contiguous()
    activations = activations.contiguous()
    return w4a16_cuda_ext.forward(W, b, S, Z, activations, group_size)


# ==============================================================================
# 2. Triton W4A16 GEMV Kernel (Safe Vector Optimized, No Atomics)
# ==============================================================================
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
    # One block maps to exactly one Output Feature row
    actual_of = tl.program_id(axis=0)
    
    # NEW Interleaved Packing Math!
    pid_of_packed = actual_of // 2
    pid_half = actual_of % 2

    acc = 0.0
    
    # Loop horizontally across the columns
    for i in range(0, IF, BLOCK_SIZE_IF):
        offsets_if = i + tl.arange(0, BLOCK_SIZE_IF)
        mask_if = offsets_if < IF
        
        # Load weights
        w_ptr = W_ptr + pid_of_packed * W_stride_0 + offsets_if * W_stride_1
        W = tl.load(w_ptr, mask=mask_if, other=0)
        
        # Load scales and zeros
        g_id = offsets_if // group_size
        S = tl.load(S_ptr + actual_of * S_stride_0 + g_id * S_stride_1, mask=mask_if, other=1.0)
        Z = tl.load(Z_ptr + actual_of * Z_stride_0 + g_id * Z_stride_1, mask=mask_if, other=0.0)
        
        # Unpack 4-bit to FP16
        w_4bit = tl.where(pid_half == 0, W & 0xF, (W >> 4) & 0xF)
        w_fp16 = (w_4bit.to(tl.float32) - Z.to(tl.float32)) * S.to(tl.float32)
        
        # Load activation and perform vector dot product via sum
        act = tl.load(act_ptr + offsets_if * act_stride_0, mask=mask_if, other=0.0)
        acc += tl.sum(w_fp16 * act.to(tl.float32))

    # Add bias and store (No atomic add needed!)
    bias = tl.load(b_ptr + actual_of)
    acc += bias.to(tl.float32)
    tl.store(OUT_ptr + actual_of * OUT_stride_0, acc.to(tl.float16))

def w4a16_gemv(W, b, S, Z, group_size, activations):
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
def unpacking_layer(W_qp: torch.Tensor):
    w0 = W_qp & 0x0F
    w1 = (W_qp >> 4) & 0x0F
    
    # NEW: We zip the Even (w0) and Odd (w1) rows together!
    # torch.stack creates alternating pairs, view flattens it back to a 2D matrix.
    return torch.stack([w0, w1], dim=1).view(-1, W_qp.shape[1])

def dequantize_layer(W_q: torch.Tensor, S: torch.Tensor, Z: torch.Tensor, b_q: torch.Tensor, group_size: int):
    N, K = W_q.shape
    W_q_reshaped = W_q.view(N, K // group_size, group_size)
    S_expanded = S.unsqueeze(-1)
    Z_expanded = Z.unsqueeze(-1)
    
    W_deq_reshaped = (W_q_reshaped.to(torch.float16) - Z_expanded) * S_expanded
    return W_deq_reshaped.view(N, K), b_q

@torch.compile
def torch_w4a16(W, b, S, Z, group_size, activations):
    activations = activations.to(torch.float16)
    W_unpacked = unpacking_layer(W)
    W_deq, b_deq = dequantize_layer(W_unpacked, S, Z, b, group_size)
    W_deq = W_deq.to(torch.float16)
    out = torch.matmul(W_deq, activations)
    if b_deq is not None:
        out += b_deq[:, None]
    return out


# ==============================================================================
# 4. Testing & Benchmarking Suite (Locked for B=1)
# ==============================================================================
def test_w4a16(shapes: tuple, atol=1e-2, rtol=1e-1, device=DEVICE):
    torch.manual_seed(0)
    OF_actual, IF, B = shapes
    assert B == 1, "Testing GEMV explicitly requires B=1"
    
    OF_packed = OF_actual // 2
    group_size = 16

    W_packed = torch.randint(0, 256, (OF_packed, IF), device=device, dtype=torch.uint8)
    b = torch.randn((OF_actual,), device=device, dtype=torch.float16)
    
    L = max(64, IF // group_size)
    S = torch.ones((OF_actual, L), device=device, dtype=torch.float16)
    Z = torch.zeros_like(S)

    activations = torch.randn((IF, B), device=device, dtype=torch.float16)

    out_triton = w4a16_gemv(W_packed, b, S, Z, group_size, activations)
    out_ref = torch_w4a16(W_packed, b, S, Z, group_size, activations)
    out_cuda = raw_cuda_w4a16(W_packed, b, S, Z, group_size, activations)

    torch.testing.assert_close(out_triton, out_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_cuda, out_ref, atol=atol, rtol=rtol)
    print("w4a16 GEMV unit test PASSED")

def benchmark_w4a16(shapes: tuple, device=DEVICE):
    torch.manual_seed(0)
    OF_actual, IF, B = shapes
    OF_packed = OF_actual // 2
    group_size = 16

    W_packed = torch.randint(0, 256, (OF_packed, IF), device=device, dtype=torch.uint8).contiguous()
    b = torch.randn((OF_actual,), device=device, dtype=torch.float16).contiguous()
    
    L = max(64, IF // group_size)
    S = torch.ones((OF_actual, L), device=device, dtype=torch.float16).contiguous()
    Z = torch.zeros_like(S).contiguous()
    activations = torch.randn((IF, B), device=device, dtype=torch.float16).contiguous()

    W_ref_fp16 = torch.randn((OF_actual, IF), device=device, dtype=torch.float16).contiguous()

    quantiles = [0.5, 0.05, 0.95]

    tri_ms, tri_min_ms, tri_max_ms = triton.testing.do_bench(
        lambda: w4a16_gemv(W_packed, b, S, Z, group_size, activations),
        quantiles=quantiles,
    )

    def torch_ref():
        return torch.matmul(W_ref_fp16, activations) + b[:, None]

    torch_ms, torch_min_ms, torch_max_ms = triton.testing.do_bench(
        torch_ref,
        quantiles=quantiles,
    )

    cuda_ms, cuda_min_ms, cuda_max_ms = triton.testing.do_bench(
        lambda: raw_cuda_w4a16(W_packed, b, S, Z, group_size, activations),
        quantiles=quantiles,
    )

    print(f"Shape: OF={OF_actual}, IF={IF}, B={B}")
    print(f"Triton W4A16: {tri_ms:.3f}ms (min: {tri_min_ms:.3f}ms)")
    print(f"PyTorch FP16: {torch_ms:.3f}ms (min: {torch_min_ms:.3f}ms)")
    print(f"Raw CUDA W4A16: {cuda_ms:.3f}ms (min: {cuda_min_ms:.3f}ms)")

    return {
        'triton_w4a16': (tri_ms, tri_min_ms, tri_max_ms),
        'torch_matmul_bias': (torch_ms, torch_min_ms, torch_max_ms),
        'raw_cuda_w4a16': (cuda_ms, cuda_min_ms, cuda_max_ms),
    }

if __name__ == "__main__":
    print("Running w4a16 correctness test...")
    test_w4a16(shapes=(1024, 1024, 1), device=DEVICE)

    print("\nRunning w4a16 benchmarks and generating plots...")
    OF_vals = [4096, 8192, 16384, 32768]
    IF = 8192
    B = 1
    records = []
    
    for OF in OF_vals:
        shapes = (OF, IF, B)
        print(f"\nBenchmarking shapes OF={OF}, IF={IF}, B={B}")
        res = benchmark_w4a16(shapes, device=DEVICE)

        flops = 2.0 * OF * IF * B
        
        for provider, key in [
            ("triton_w4a16", "triton_w4a16"),
            ("torch_matmul_bias", "torch_matmul_bias"),
            ("raw_cuda_w4a16", "raw_cuda_w4a16")
        ]:
            ms, min_ms, max_ms = res[key]
            tflops = flops * 1e-12 / (ms * 1e-3)
            
            records.append({
                "OF": OF, "IF": IF, "B": B, "provider": provider,
                "ms": ms, "tflops": tflops,
            })

    df = pd.DataFrame.from_records(records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = f"w4a16_gemv_benchmark_{timestamp}.csv"
    df.to_csv(results_csv, index=False)
    print(f"\nSaved benchmark data to {results_csv}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    plot_configs = [
        ("triton_w4a16", "Triton W4A16", "tab:blue"),
        ("torch_matmul_bias", "PyTorch matmul+bias", "tab:orange"),
        ("raw_cuda_w4a16", "Raw CUDA W4A16", "tab:green")
    ]

    for provider, label, color in plot_configs:
        sub = df[df["provider"] == provider]
        ax1.plot(sub["OF"], sub["tflops"], marker="o", label=label, color=color)

    ax1.set_xlabel("Output features (OF)")
    ax1.set_ylabel("TFLOPS (approximate)")
    ax1.set_title("Throughput: W4A16 GEMV vs PyTorch FP16")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    for provider, label, color in plot_configs:
        sub = df[df["provider"] == provider]
        ax2.plot(sub["OF"], sub["ms"], marker="o", label=label, color=color)

    ax2.set_xlabel("Output features (OF)")
    ax2.set_ylabel("Execution Time (ms)")
    ax2.set_title("Latency: W4A16 GEMV vs PyTorch FP16")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plot_path = f"w4a16_benchmark_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved performance plots to {plot_path}")