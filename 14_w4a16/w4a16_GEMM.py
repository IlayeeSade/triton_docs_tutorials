import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from datetime import datetime
import argparse
import math

DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

########## ALGORITHM (1) ###########

autotune_configs_w4a16 = [
    triton.Config({'BLOCK_SIZE_OF': 64, 'BLOCK_SIZE_IF': 64, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=4),
    triton.Config({'BLOCK_SIZE_OF': 64, 'BLOCK_SIZE_IF': 32, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=4),
    triton.Config({'BLOCK_SIZE_OF': 64, 'BLOCK_SIZE_IF': 32, 'BLOCK_SIZE_B': 128, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=4),
    triton.Config({'BLOCK_SIZE_OF': 32, 'BLOCK_SIZE_IF': 32, 'BLOCK_SIZE_B': 128, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=4),
    triton.Config({'BLOCK_SIZE_OF': 128, 'BLOCK_SIZE_IF': 32, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=4),
    triton.Config({'BLOCK_SIZE_OF': 128, 'BLOCK_SIZE_IF': 64, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=8),
    triton.Config({'BLOCK_SIZE_OF': 256, 'BLOCK_SIZE_IF': 64, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=8),
]

@triton.autotune(configs=autotune_configs_w4a16, key=['OF', 'IF', 'B'])
@triton.jit
def _w4a16_kernel(
    W_ptr, b_ptr, S_ptr, Z_ptr, activations_ptr, OUT_ptr,
    W_stride_0, W_stride_1,
    b_stride_0, b_stride_1,
    S_stride_0, S_stride_1,
    Z_stride_0, Z_stride_1,
    OF, IF, B, group_size, OFG,
    activations_stride_0, activations_stride_1,
    OUT_stride_0, OUT_stride_1,
    BLOCK_SIZE_OF: tl.constexpr, BLOCK_SIZE_IF: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
    GROUP_SIZE: tl.constexpr, num_stages: tl.constexpr,
):
    # raw_PID_OF = tl.program_id(axis=0)
    # raw_PID_B = tl.program_id(axis=1)

    # num_PID_along_OF = tl.cdiv(OF, BLOCK_SIZE_OF)
    # num_PID_along_B = tl.cdiv(B, BLOCK_SIZE_B)

    # PID_half = raw_PID_OF // num_PID_along_OF
    # logical_PID_OF = raw_PID_OF % num_PID_along_OF

    # PID = raw_PID_B * num_PID_along_OF + logical_PID_OF

    # # Group-major ordering
    # num_PID_in_group = GROUP_SIZE * num_PID_along_B
    # group_id = PID // num_PID_in_group
    # first_PID_in_group_along_OF = group_id * GROUP_SIZE
    # group_size_adj = min(num_PID_along_OF - first_PID_in_group_along_OF, GROUP_SIZE)

    # PID_OF = first_PID_in_group_along_OF + ((PID % num_PID_in_group) % group_size_adj)
    # PID_B = (PID % num_PID_in_group) // group_size_adj

    #for some reason the former technique works better emprirically
    #although this is modifies swizzling
    Raw_PID = tl.program_id(axis=0)
    num_PID_along_OF = tl.cdiv(OF, BLOCK_SIZE_OF)
    num_PID_along_B = tl.cdiv(B, BLOCK_SIZE_B)

    # we are making the corresponding PID in the other half be right afterwards
    PID_half = Raw_PID % 2
    PID = Raw_PID // 2

    num_PID_in_group = GROUP_SIZE * num_PID_along_B
    group_id = PID // num_PID_in_group 
    
    first_PID_in_group_along_OF = group_id * GROUP_SIZE 
    group_size_adj = min(num_PID_along_OF - first_PID_in_group_along_OF, GROUP_SIZE) 
    
    PID_OF = first_PID_in_group_along_OF + (PID % group_size_adj)
    PID_B = (PID % num_PID_in_group) // group_size_adj



    offsets_OF = PID_OF * BLOCK_SIZE_OF + tl.arange(0, BLOCK_SIZE_OF)
    offsets_B = PID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offsets_IF = tl.arange(0, BLOCK_SIZE_IF)

    a_offsets = offsets_OF[:, None] * W_stride_0 + offsets_IF[None, :] * W_stride_1
    b_offsets = offsets_IF[:, None] * activations_stride_0 + offsets_B[None, :] * activations_stride_1

    offsets_OF_G_adj = offsets_OF + (PID_half * OF)

    accumulator = tl.zeros((BLOCK_SIZE_OF, BLOCK_SIZE_B), dtype=tl.float32)

    for step in range(0, tl.cdiv(IF, BLOCK_SIZE_IF)):
        # repeating pointers because some are the same
        offsets_IF_G_adj = offsets_IF // group_size

        s_offsets = offsets_OF_G_adj[:, None] * S_stride_0 + offsets_IF_G_adj[None, :] * S_stride_1
        z_offsets = offsets_OF_G_adj[:, None] * Z_stride_0 + offsets_IF_G_adj[None, :] * Z_stride_1

        mask = offsets_IF < IF

        w_mask = (offsets_OF[:, None] < OF) & mask[None, :]
        act_mask = mask[:, None] & (offsets_B[None, :] < B)
        sz_mask = (offsets_OF_G_adj[:, None] < (OF * 2)) & (offsets_IF_G_adj[None, :] < (IF // group_size))

        S = tl.load(S_ptr + s_offsets, mask=sz_mask, other=0.0)
        Z = tl.load(Z_ptr + z_offsets, mask=sz_mask, other=0.0)
        W = tl.load(W_ptr + a_offsets, mask=w_mask, other=0.0)
        activations = tl.load(activations_ptr + b_offsets, mask=act_mask, other=0.0)

        W_h = tl.where(PID_half == 0, W & 0x0F, (W >> 4) & 0x0F)
        W_h = W_h.to(tl.float16)
        W_h -= Z
        W_h *= S

        accumulator = tl.dot(W_h, activations, acc=accumulator)

        a_offsets += BLOCK_SIZE_IF * W_stride_1
        b_offsets += BLOCK_SIZE_IF * activations_stride_0
        offsets_IF += BLOCK_SIZE_IF

    # output masks check against 2 * OF
    b_offsets = offsets_OF_G_adj * b_stride_0
    b_mask = offsets_OF_G_adj < (OF * 2)
    b_vals = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)

    accumulator += b_vals[:, None]
    accumulator = accumulator.to(tl.float16)

    c_offsets = offsets_OF_G_adj[:, None] * OUT_stride_0 + offsets_B[None, :] * OUT_stride_1
    c_mask = (offsets_OF_G_adj[:, None] < (OF * 2)) & (offsets_B[None, :] < B)
    tl.store(OUT_ptr + c_offsets, accumulator, mask=c_mask)


def w4a16(W, b, S, Z, group_size, activations):
    # assertions
    assert W.shape[1] == activations.shape[0], "incompatible dimensions"

    (OF, IF), (_, B), (_, OFG) = W.shape, activations.shape, S.shape

    OUT = torch.zeros((2 * OF, B), device=W.device, dtype=torch.float16)

    grid = lambda meta: ((2 * triton.cdiv(OF, meta['BLOCK_SIZE_OF'])) * triton.cdiv(B, meta['BLOCK_SIZE_B']), )
    _w4a16_kernel[grid](
        W, b, S, Z, activations, OUT,
        W.stride(0), W.stride(1),
        b.stride(0), 0,  # b is treated as 1D; kernel never uses b_stride_1
        S.stride(0), S.stride(1),
        Z.stride(0), Z.stride(1),
        OF, IF, B, group_size, OFG,
        activations.stride(0), activations.stride(1),
        OUT.stride(0), OUT.stride(1),
    )
    return OUT


def unpacking_layer(W_qp: torch.Tensor):
    """
    Unpacks a packed 8-bit weight tensor back into two 4-bit values (as uint8).
    Assumes packing is along the Output Features (OF / N) dimension.
    """
    # Extract lower 4 bits (First half of output channels)
    w0 = W_qp & 0x0F
    # Extract upper 4 bits and shift right (Second half of output channels)
    w1 = (W_qp >> 4) & 0x0F

    # Concatenate along the N (Output Features) dimension (dim=0)
    wF = torch.cat([w0, w1], dim=0)
    return wF


def dequantize_layer(W_q: torch.Tensor, S: torch.Tensor, Z: torch.Tensor, b_q: torch.Tensor, group_size: int):
    """
    De-quantizes 4-bit weights back to FP16 and returns the FP16 bias.
    """
    N, K = W_q.shape

    W_q_reshaped = W_q.view(N, K // group_size, group_size)
    S_expanded = S.unsqueeze(-1)
    Z_expanded = Z.unsqueeze(-1)

    # Perform de-quantization math strictly in FP16
    W_deq_reshaped = (W_q_reshaped.to(torch.float16) - Z_expanded) * S_expanded
    W_deq = W_deq_reshaped.view(N, K)

    return W_deq, b_q


@torch.compile
def torch_w4a16(W, b, S, Z, group_size, activations):
    # W comes in as (OF_packed, IF) -> unpacked to (2*OF_packed, IF)
    activations = activations.to(torch.float16)

    W_unpacked = unpacking_layer(W)
    W_deq, b_deq = dequantize_layer(W_unpacked, S, Z, b, group_size)

    W_deq = W_deq.to(torch.float16)

    # We use torch.matmul instead of F.linear because activations is (IF, B), not (B, IF)
    out = torch.matmul(W_deq, activations)

    if b_deq is not None:
        b_deq = b_deq.to(torch.float16)
        out += b_deq[:, None]

    return out


def test_w4a16(shapes: tuple, atol=1e-2, rtol=1e-1, device=DEVICE):
    """
    Unit test for the Triton w4a16 kernel.
    """
    torch.manual_seed(0)
    assert isinstance(shapes, tuple) and len(shapes) == 3
    
    # OF is the ACTUAL number of output features we want (e.g., 1024)
    OF_actual, IF, B = shapes
    
    # Since weights are packed 2-to-1 along the OF axis, the physical tensor has half the rows
    OF_packed = OF_actual // 2
    group_size = 16

    # 1. Construct standard 8-bit packed weights (uint8, not int16)
    W_packed = torch.randint(0, 256, (OF_packed, IF), device=device, dtype=torch.uint8)

    # 2. Bias, Scales, and Zeros map to the ACTUAL output features
    b = torch.randn((OF_actual,), device=device, dtype=torch.float16)
    
    L = max(64, IF // group_size)
    S = torch.ones((OF_actual, L), device=device, dtype=torch.float16)
    Z = torch.zeros_like(S)

    # 3. Activations: (IF, B)
    activations = torch.randn((IF, B), device=device, dtype=torch.float16)

    # 4. Triton kernel output
    out_triton = w4a16(W_packed, b, S, Z, group_size, activations)

    # 5. Reference PyTorch output using the unpacked layer logic
    out_ref = torch_w4a16(W_packed, b, S, Z, group_size, activations)

    torch.testing.assert_close(out_triton, out_ref, atol=atol, rtol=rtol)
    print("w4a16 unit test PASSED")


def benchmark_w4a16(shapes: tuple, device=DEVICE):
    """
    Benchmark the Triton w4a16 kernel against PyTorch matmul + bias.
    """
    torch.manual_seed(0)
    assert isinstance(shapes, tuple) and len(shapes) == 3
    
    OF_actual, IF, B = shapes
    OF_packed = OF_actual // 2
    group_size = 16

    # Initialize correct shapes and uint8
    W_packed = torch.randint(0, 256, (OF_packed, IF), device=device, dtype=torch.uint8).contiguous()
    b = torch.randn((OF_actual,), device=device, dtype=torch.float16).contiguous()
    
    L = max(64, IF // group_size)
    S = torch.ones((OF_actual, L), device=device, dtype=torch.float16).contiguous()
    Z = torch.zeros_like(S).contiguous()
    activations = torch.randn((IF, B), device=device, dtype=torch.float16).contiguous()

    # For the PyTorch baseline, we want to measure a raw FP16 matmul, 
    # since torch_w4a16 dequantization is too slow to be a fair baseline.
    W_ref_fp16 = torch.randn((OF_actual, IF), device=device, dtype=torch.float16).contiguous()

    quantiles = [0.5, 0.05, 0.95]

    # Triton kernel benchmark
    tri_ms, tri_min_ms, tri_max_ms = triton.testing.do_bench(
        lambda: w4a16(W_packed, b, S, Z, group_size, activations),
        quantiles=quantiles,
    )

    # PyTorch reference benchmark: matmul + bias
    def torch_ref():
        return torch.matmul(W_ref_fp16, activations) + b[:, None]

    torch_ms, torch_min_ms, torch_max_ms = triton.testing.do_bench(
        torch_ref,
        quantiles=quantiles,
    )

    print(f"Shape: OF={OF_actual}, IF={IF}, B={B}")
    print(f"Triton W4A16: {tri_ms:.3f}ms (min: {tri_min_ms:.3f}ms, max: {tri_max_ms:.3f}ms)")
    print(f"PyTorch FP16 matmul+bias: {torch_ms:.3f}ms (min: {torch_min_ms:.3f}ms, max: {torch_max_ms:.3f}ms)")

    speedup_w4a16 = torch_ms / tri_ms
    print(f"Speedup vs PyTorch FP16: {speedup_w4a16:.2f}x")

    return {
        'triton_w4a16': (tri_ms, tri_min_ms, tri_max_ms),
        'torch_matmul_bias': (torch_ms, torch_min_ms, torch_max_ms),
    }


if __name__ == "__main__":
    # 1) Basic correctness check
    print("Running w4a16 correctness test...")
    test_w4a16(shapes=(1024, 1024, 256), device=DEVICE)

    # 2) Benchmark sweep
    print("Running w4a16 benchmarks and generating plots...")
    OF_vals = [4096, 8192, 16384, 32768]
    IF = 8192
    B = 256
    records = []
    for OF in OF_vals:
        shapes = (OF, IF, B)
        print(f"\nBenchmarking shapes OF={OF}, IF={IF}, B={B}")
        res = benchmark_w4a16(shapes, device=DEVICE)

        tri_ms, tri_min_ms, tri_max_ms = res["triton_w4a16"]
        torch_ms, torch_min_ms, torch_max_ms = res["torch_matmul_bias"]

        flops = 2.0 * OF * IF * B
        tri_tflops = flops * 1e-12 / (tri_ms * 1e-3)
        torch_tflops = flops * 1e-12 / (torch_ms * 1e-3)

        records.append({
            "OF": OF, "IF": IF, "B": B, "provider": "triton_w4a16",
            "ms": tri_ms, "tflops": tri_tflops,
        })
        records.append({
            "OF": OF, "IF": IF, "B": B, "provider": "torch_matmul_bias",
            "ms": torch_ms, "tflops": torch_tflops,
        })

    df = pd.DataFrame.from_records(records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = f"w4a16_gemm_benchmark_{timestamp}.csv"
    df.to_csv(results_csv, index=False)
    print(f"\nSaved benchmark data to {results_csv}")

    # Create a figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: TFLOPS vs OF
    for provider, label, color in [
        ("triton_w4a16", "Triton W4A16", "tab:blue"),
        ("torch_matmul_bias", "PyTorch matmul+bias", "tab:orange"),
    ]:
        sub = df[df["provider"] == provider]
        ax1.plot(sub["OF"], sub["tflops"], marker="o", label=label, color=color)

    ax1.set_xlabel("Output features (OF)")
    ax1.set_ylabel("TFLOPS (approximate)")
    ax1.set_title("Throughput: W4A16 GEMM vs PyTorch FP16")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Execution Time (ms) vs OF
    for provider, label, color in [
        ("triton_w4a16", "Triton W4A16", "tab:blue"),
        ("torch_matmul_bias", "PyTorch matmul+bias", "tab:orange"),
    ]:
        sub = df[df["provider"] == provider]
        ax2.plot(sub["OF"], sub["ms"], marker="o", label=label, color=color)

    ax2.set_xlabel("Output features (OF)")
    ax2.set_ylabel("Execution Time (ms)")
    ax2.set_title("Latency: W4A16 GEMM vs PyTorch FP16")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plot_path = f"w4a16_benchmark_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved performance plots to {plot_path}")