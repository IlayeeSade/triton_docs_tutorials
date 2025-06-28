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


@triton.jit
def _superlinear_kernel(
    x_ptr,
    w1_ptr,
    b1_ptr,
    out_ptr,
    B, D, M, H,
    stride_xb, stride_xd, stride_xm,
    stride_wm, stride_wh, stride_wd,
    stride_bb, stride_bd, stride_bh,
    stride_ob, stride_od, stride_oh,
):
    # Get program IDs
    pid_b = tl.program_id(axis=0)  # batch dimension
    pid_d = tl.program_id(axis=1)  # d_model dimension  
    pid_h = tl.program_id(axis=2)  # hidden dimension
    
    # Check bounds
    if pid_b >= B or pid_d >= D or pid_h >= H:
        return
    
    # Load x slice: (B, D, M) -> get (M,) for this (b, d) position
    x_ptrs = x_ptr + (pid_b * stride_xb + pid_d * stride_xd + tl.arange(0, M) * stride_xm)
    x = tl.load(x_ptrs)
    
    # Load w1 slice: (M, H, D) -> get (M,) for this (h, d) position  
    w1_ptrs = w1_ptr + (tl.arange(0, M) * stride_wm + pid_h * stride_wh + pid_d * stride_wd)
    w1 = tl.load(w1_ptrs)
    
    # Load b1: (1, D, H) -> get scalar for this (d, h) position
    b1_ptrs = b1_ptr + (0 * stride_bb + pid_d * stride_bd + pid_h * stride_bh)
    b1 = tl.load(b1_ptrs)
    
    # Compute dot product: sum(x * w1) + b1
    acc = tl.dot(x, w1) + b1
    
    # Store result
    out_ptrs = out_ptr + (pid_b * stride_ob + pid_d * stride_od + pid_h * stride_oh)
    tl.store(out_ptrs, acc)

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
    
    grid = (B, D, H)
    _superlinear_kernel[grid](
        x, w1, b1, O,
        B, D, M, H, 
        x.stride(0), x.stride(1), x.stride(2),
        w1.stride(0), w1.stride(1), w1.stride(2),
        b1.stride(0), b1.stride(1), b1.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
    )
    
    return O

    
    def test_superlinear():
        """Test function to compare einsum implementation with Triton kernel."""
        print("Testing SuperLinear kernel vs einsum implementation...")
        
        # Test parameters
        B, D, M, H = 2, 64, 128, 1  # batch, neurons, memory_length, hidden_dim
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
        
        # Compare results
        diff = torch.abs(out_einsum - out_kernel).max().item()
        print(f"Max difference: {diff:.6f}")
        
        if diff < 1e-4:
            print("✅ Test passed: Kernel matches einsum implementation")
        else:
            print("❌ Test failed: Kernel differs from einsum implementation")
            print(f"Einsum output shape: {out_einsum.shape}")
            print(f"Kernel output shape: {out_kernel.shape}")
        
        return diff < 1e-4
    
    def benchmark_superlinear():
        """Benchmark SuperLinear kernel vs einsum implementation."""
        print("\nBenchmarking SuperLinear kernel vs einsum implementation...")
        
        # Test configurations with different shapes
        configs = [
            # Small scale (typical for small models)
            (1, 32, 64, 1),
            (2, 64, 128, 1),
            
            # Medium scale (typical for medium models)
            (4, 128, 256, 1),
            (8, 256, 512, 1),
            
            # Large scale (typical for large models)
            (16, 512, 1024, 1),
            (32, 1024, 2048, 1),
            
            # Extreme scale (for testing limits)
            (64, 2048, 4096, 1),
        ]
        
        results = []
        
        for B, D, M, H in configs:
            print(f"\n{'='*60}")
            print(f"Testing: B={B}, D={D}, M={M}, H={H}")
            print(f"Total elements: {B*D*M:,} (input), {B*D*H:,} (output)")
            print(f"Memory usage: ~{B*D*M*4/1024/1024:.1f}MB (input), ~{B*D*H*4/1024/1024:.1f}MB (output)")
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
        
        if test_passed:
            # Run benchmarks
            benchmark_results = benchmark_superlinear()
            
            # Print summary
            print("\n" + "="*80)
            print("BENCHMARK SUMMARY")
            print("="*80)
            print(f"{'Shape':<20} {'Elements':<12} {'Memory':<8} {'Speedup':<10}")
            print("-"*80)
            for result in benchmark_results:
                B, D, M, H = result['shape']
                elements = result['total_elements']
                memory = result['memory_mb']
                speedup = result['speedup']
                print(f"{f'B={B},D={D},M={M}':<20} {elements:<12,} {memory:<8.1f}MB {speedup:<10.2f}x")
        else:
            print("Skipping benchmarks due to test failure")


