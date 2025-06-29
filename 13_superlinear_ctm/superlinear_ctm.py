import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from datetime import datetime
import argparse

import torch
DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

autotune_configs = [
    triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_D': 64, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_H': 64, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=4),
    triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_D': 32, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_H': 32, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=4),
    triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_D': 64, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_H': 128, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=8),
    triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_D': 32, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_H': 64, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=8),
]

@triton.autotune(configs=autotune_configs, key=['B', 'D', 'M', 'H'])
@triton.jit
def _superlinear_kernel(
    x_ptr,
    w1_ptr,
    b1_ptr,
    out_ptr,
    B, D, M, H,
    stride_xb, stride_xd, stride_xm,
    stride_wm, stride_wh, stride_wd,
    stride_bd, stride_bh,
    stride_ob, stride_od, stride_oh,
    BLOCK_SIZE_B : tl.constexpr,
    BLOCK_SIZE_D : tl.constexpr,
    BLOCK_SIZE_M : tl.constexpr,
    BLOCK_SIZE_H : tl.constexpr,
    GROUP_SIZE : tl.constexpr,
    num_stages : tl.constexpr,
    
):
    # Get program IDs - each program handles one (B_block, D, H_block) tile
    pid_b = tl.program_id(axis=0)  
    pid_d = tl.program_id(axis=1)  
    pid_h = tl.program_id(axis=2)
    # Superlinear part 1 ---------------- (LAYERNORM CALCULATION)
    # We need an 2 (M, ) accumulators one for mean and for std.
    # We want to sum over each kernel b_m, each also have a designated neuron, pid_d
    # And a disignated pid_b which decides our batch to sum over,
    # So we sum over our b_m in b_b in index d and add it to the shared acc of the
    # programs index d, we will have an acc for each index d and we will sum it like a tournament

    offs_b = pid_b + tl.arange(0, BLOCK_SIZE_B)
    offs_m = tl.arange(0, M)

    mu = tl.zeros((BLOCK_SIZE_B,), dtype=tl.float32)
    std = tl.zeros((BLOCK_SIZE_B,), dtype=tl.float32)
    placeholder = tl.zeros((BLOCK_SIZE_B, M), dtype=tl.float32)

    placeholder = tl.load(
        x_ptr + (
            offs_b * stride_xb + 
            pid_d * stride_xd +
            offs_m * stride_xm
        ),
        mask = (offs_b < B) & (pid_d < D) & (offs_m < M),
        other = 0.0
    )
    
    mu = tl.sum(
        placeholder,
        axis=1
    ) / M

    placeholder = placeholder - mu

    std = tl.sqrt(
        tl.sum(
            placeholder * placeholder,
            axis=1
        ) / M
        
    )

    # Superlinear part 2 ---------------- (SUPERLINEAR CALCULATION)  
    
    # Calculate actual indices
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    # Boundary checks
    mask_b = offs_b < B
    mask_h = offs_h < H

    # Initialize accumulator with correct precision
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=tl.float32)

    # Loop over M dimension - this is the reduction dimension
    for m_start in range(0, M, BLOCK_SIZE_M):
        offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offs_m < M

        # Load x: shape (B, D, M) -> need x[offs_b, pid_d, offs_m] -> (B_B, B_M)
        x_ptrs = x_ptr + (offs_b[:, None] * stride_xb + 
                         pid_d * stride_xd + 
                         offs_m[None, :] * stride_xm)
        x_block = tl.load(x_ptrs, mask=mask_b[:, None] & mask_m[None, :], other=0.0)
        x_block = (x_block - mu[:, None]) / std[:, None] 
        # Load w1: shape (M, H, D) -> need w1[offs_m, offs_h, pid_d] -> (B_M, B_H)
        w1_ptrs = w1_ptr + (offs_m[:, None] * stride_wm + 
                           offs_h[None, :] * stride_wh + 
                           pid_d * stride_wd)
        w1_block = tl.load(w1_ptrs, mask=mask_m[:, None] & mask_h[None, :], other=0.0)
        
        # Allow tf32 causes less precision but more speed up
        acc = tl.dot(x_block, w1_block, allow_tf32=False, acc=acc)
    
    # Add bias: b1 shape is (1, D, H) -> need b1[0, pid_d, offs_h]
    b1_ptrs = b1_ptr + (pid_d * stride_bd + offs_h * stride_bh)
    bias = tl.load(b1_ptrs, mask=mask_h, other=0.0)
    
    # Add bias (broadcast across batch dimension)
    acc = acc + bias[None, :]
        
    # Store result: out shape (B, D, H) -> store at out[offs_b, pid_d, offs_h]
    out_ptrs = out_ptr + (offs_b[:, None] * stride_ob + 
                         pid_d * stride_od + 
                         offs_h[None, :] * stride_oh)
    tl.store(out_ptrs, acc, mask=mask_b[:, None] & mask_h[None, :])


def superlinear(
        x, w1, b1,
        T=1.0,
        do_norm=False,
        dropout=0):
    # x: (B, D, M) where D=d_model=N neurons in CTM, M=history/memory length
    # w1: (M, H, D)
    # b1: (1, D, H)
    # einsum result: (B, D, H)

    (B, D, M), (_, H, _) = x.shape, w1.shape

    # Assertions for tensor shape compatibility
    assert w1.shape[0] == x.shape[2], f"history-length mismatch: w1.shape[0]={w1.shape[0]} vs x.shape[2]={x.shape[2]}"
    assert w1.shape[2] == x.shape[1], f"neuron-nb mismatch: w1.shape[2]={w1.shape[2]} vs x.shape[1]={x.shape[1]}"
    assert b1.shape[0] == 1, f"b1 first dimension should be 1, got {b1.shape[0]}"
    assert b1.shape[1] == x.shape[1], f"b1 second dimension mismatch: b1.shape[1]={b1.shape[1]} vs x.shape[1]={x.shape[1]}"
    assert b1.shape[2] == w1.shape[1], f"b1 third dimension mismatch: b1.shape[2]={b1.shape[2]} vs w1.shape[1]={w1.shape[1]}"
    
    # Assertions for parameter validity
    assert T > 0, f"Temperature T must be positive, got {T}"
    assert 0 <= dropout <= 1, f"Dropout must be between 0 and 1, got {dropout}"
    
    # Assertions for tensor dimensions
    assert len(x.shape) == 3, f"x should be 3D tensor, got shape {x.shape}"
    assert len(w1.shape) == 3, f"w1 should be 3D tensor, got shape {w1.shape}"
    assert len(b1.shape) == 3, f"b1 should be 3D tensor, got shape {b1.shape}"

    O = torch.empty((B, D, H), device=DEVICE, dtype=torch.float32)
    
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']), D, triton.cdiv(H, meta['BLOCK_SIZE_H']))
    _superlinear_kernel[grid](
        x, w1, b1, O,
        B, D, M, H, 
        x.stride(0), x.stride(1), x.stride(2),
        w1.stride(0), w1.stride(1), w1.stride(2),
    b1.stride(1), b1.stride(2),  # b1.stride(0) is skipped since first dim is 1
        O.stride(0), O.stride(1), O.stride(2),
    )
    
    return O

    
def test_superlinear():
    """Test function to compare einsum implementation with Triton kernel."""
    print("Testing SuperLinear kernel vs einsum implementation...")
    
    # Test parameters with varying H dimensions
    test_configs = [
        (2, 64, 128, 32),   # batch, neurons, memory_length, hidden_dim
        (4, 32, 64, 128),   # Different H sizes
        (1, 128, 256, 64),  # Single batch
        (8, 16, 32, 256),   # Large H
    ]
    
    all_passed = True
    
    for i, (B, D, M, H) in enumerate(test_configs):
        print(f"\nTest {i+1}: B={B}, D={D}, M={M}, H={H}")
        T = 2.0
        
        # Create test tensors
        x = torch.randn(B, D, M, device=DEVICE, dtype=torch.float32)
        w1 = torch.randn(M, H, D, device=DEVICE, dtype=torch.float32)
        b1 = torch.randn(1, D, H, device=DEVICE, dtype=torch.float32)
        
        # Einsum implementation (reference)
        def einsum_superlinear(x, w1, b1, T):
            return torch.einsum('BDM,MHD->BDH', x, w1) + b1
        
        # Run both implementations
        out_einsum = einsum_superlinear(x, w1, b1, T)
        out_kernel = superlinear(x, w1, b1, T)
        
        # Compare results with more detailed analysis
        diff = torch.abs(out_einsum - out_kernel).max().item()
        mean_diff = torch.abs(out_einsum - out_kernel).mean().item()
        rel_error = (torch.abs(out_einsum - out_kernel) / (torch.abs(out_einsum) + 1e-8)).max().item()
        
        print(f"  Max difference: {diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        print(f"  Max relative error: {rel_error:.6f}")
        
        # Use a more reasonable tolerance for floating point comparisons
        tolerance = 1e-3  # Relaxed tolerance for float32 precision
        
        if diff < tolerance:
            print("  ✅ Test passed: Kernel matches einsum implementation")
        else:
            print("  ❌ Test failed: Kernel differs from einsum implementation")
            print(f"  Einsum output shape: {out_einsum.shape}")
            print(f"  Kernel output shape: {out_kernel.shape}")
            
            # Debug: Print some values for comparison
            print(f"  Einsum sample values: {out_einsum[0, 0, :min(5,H)]}")
            print(f"  Kernel sample values: {out_kernel[0, 0, :min(5,H)]}")
            
            # Check if it's a systematic bias or random error
            bias_check = (out_einsum - out_kernel).mean().item()
            print(f"  Systematic bias: {bias_check:.6f}")
            
            all_passed = False
            
        # Update individual test result
        if diff >= tolerance:
            all_passed = False
    
    return all_passed

def benchmark_superlinear():
    """Benchmark SuperLinear kernel vs einsum implementation."""
    print("\nBenchmarking SuperLinear kernel vs einsum implementation...")
    
    # Test configurations with different shapes and varying H dimensions
    configs = [
        # Small scale - varying H
        (1, 16, 2, 8),
        (2, 32, 4, 16),
        
        # Medium scale - varying H  
        (4, 64, 8, 32),
        (8, 128, 16, 64),
        
        # Large scale - varying H
        (16, 256, 32, 128),
        (32, 512, 64, 256),
        
        # Extreme scale - varying H
        (64, 1024, 128, 512),
        (128, 2048, 256, 1024),
    ]

    
    results = []
    
    for B, D, M, H in configs:
        print(f"\n{'='*60}")
        print(f"Testing: B={B}, D={D}, M={M}, H={H}")
        print(f"Total elements: {B*D*M:,} (input), {B*D*H:,} (output)")
        print(f"Memory usage: ~{B*D*M*4/1024/1024:.1f}MB (input), ~{B*D*H*4/1024/1024:.1f}MB (output)")
        print(f"Weight tensor: {M*H*D:,} elements (~{M*H*D*4/1024/1024:.1f}MB)")
        print('='*60)
        
        # Create test tensors
        x = torch.randn(B, D, M, device=DEVICE, dtype=torch.float32)
        w1 = torch.randn(M, H, D, device=DEVICE, dtype=torch.float32)
        b1 = torch.randn(1, D, H, device=DEVICE, dtype=torch.float32)
        
        # Einsum implementation
        def einsum_superlinear(x, w1, b1, T):
            return torch.einsum('BDM,MHD->BDH', x, w1) + b1
        
        # Warmup
        print("Warming up...")
        for _ in range(10):
            einsum_superlinear(x, w1, b1, 1.0)
            superlinear(x, w1, b1, 1.0)
        
        # Benchmark
        quantiles = [0.5, 0.05, 0.95]
        
        print("Running einsum benchmark...")
        einsum_ms, einsum_min_ms, einsum_max_ms = triton.testing.do_bench(
            lambda: einsum_superlinear(x, w1, b1, 1.0),
            quantiles=quantiles
        )
        
        print("Running kernel benchmark...")
        kernel_ms, kernel_min_ms, kernel_max_ms = triton.testing.do_bench(
            lambda: superlinear(x, w1, b1, 1.0),
            quantiles=quantiles
        )
        
        speedup = einsum_ms / kernel_ms
        
        print(f"Einsum: {einsum_ms:.3f}ms (min: {einsum_min_ms:.3f}ms, max: {einsum_max_ms:.3f}ms)")
        print(f"Kernel: {kernel_ms:.3f}ms (min: {kernel_min_ms:.3f}ms, max: {kernel_max_ms:.3f}ms)")
        print(f"Speedup: {speedup:.2f}x")
        
        results.append({
            'shape': (B, D, M, H),
            'einsum': (einsum_ms, einsum_min_ms, einsum_max_ms),
            'kernel': (kernel_ms, kernel_min_ms, kernel_max_ms),
            'speedup': speedup,
            'total_elements': B*D*M,
            'memory_mb': B*D*M*4/1024/1024
        })
    
    return results

if __name__ == "__main__":
    # Run tests
    test_passed = test_superlinear()
    
    print(test_passed)
    if not test_passed:
        # Run benchmarks
        benchmark_results = benchmark_superlinear()
        
        # Print summary
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(f"{'Shape':<25} {'Elements':<12} {'Memory':<10} {'Weight MB':<10} {'Speedup':<10}")
        print("-"*90)
        for result in benchmark_results:
            B, D, M, H = result['shape']
            elements = result['total_elements']
            memory = result['memory_mb']
            weight_mb = M*H*D*4/1024/1024
            speedup = result['speedup']
            print(f"{f'B={B},D={D},M={M},H={H}':<25} {elements:<12,} {memory:<10.1f}MB {weight_mb:<10.1f}MB {speedup:<10.2f}x")
    else:
        print("Skipping benchmarks due to test failure")