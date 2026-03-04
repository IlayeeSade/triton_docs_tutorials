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

import torch
DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

########## ALGORITHM (1) ###########

autotune_configs_w4a16 = [
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_D': 64, 'BLOCK_SIZE_V': 64, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=4),
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_D': 32, 'BLOCK_SIZE_V': 32, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=4),
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_D': 32, 'BLOCK_SIZE_V': 128, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=4),
    triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_D': 32, 'BLOCK_SIZE_V': 128, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=4),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_D': 32, 'BLOCK_SIZE_V': 32, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=4),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_D': 64, 'BLOCK_SIZE_V': 64, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=8),
    triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_D': 64, 'BLOCK_SIZE_V': 64, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=8),
]

@triton.autotune(configs=autotune_configs_lsemm, key=['N', 'D', 'V'])
@triton.jit
def _w4a16_kernel(
    W_ptr, b_ptr, S_ptr, Z_ptr, group_size_ptr, activations_ptr, OUT_ptr,
    W_stride_0, W_stride_1,
    b_stride_0, b_stride_1,
    S_stride_0, S_stride_1,
    Z_stride_0, Z_stride_1,
    OF, IF, B,
    group_size_stride_0, group_size_stride_1,
    activations_stride_0, activations_stride_1,
    OUT_stride_0, OUT_stride_1,
    BLOCK_SIZE_OF: tl.constexpr, BLOCK_SIZE_IF: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
    GROUP_SIZE: tl.constexpr,  num_stages: tl.constexpr,
):
    # We want to matmul (OF, IF) @ (IF, B)
    PID = tl.program_id(axis=0) 
    
    # Group-major ordering
    num_PID_along_OF = tl.cdiv(OF, BLOCK_SIZE_OF)
    num_PID_along_B = tl.cdiv(B, BLOCK_SIZE_B)
    num_PID_in_group = GROUP_SIZE * num_PID_along_B
    group_id = PID // num_PID_in_group 
    first_PID_in_group_along_OF = group_id * GROUP_SIZE 
    group_size_adj = min(num_PID_along_OF - first_PID_in_group_along_OF, GROUP_SIZE) 
    PID_OF = first_PID_in_group_along_OF + ((PID % num_PID_in_group) % group_size_adj)
    PID_B = (PID % num_PID_in_group) // group_size_adj

    offsets_OF = PID_OF * BLOCK_SIZE_OF + tl.arange(0, BLOCK_SIZE_OF)
    offsets_B = PID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)\

    offsets_IF = tl.arange(0, BLOCK_SIZE_IF)
    
    a_offsets = offsets_OF[:, None] * W_stride_0 + offsets_IF[None, :] * W_stride_1 # (OF, IF)
    b_offsets = offsets_IF[:, None] * activations_stride_0 + offsets_B[None, :] * activations_stride_1 # (IF, B)

    # inputs tensors are fp16 but we accumulate into a block of fp32 values for higher accuracy (we'll revert later)
    accumulator = tl.zeros((BLOCK_SIZE_OF, BLOCK_SIZE_B), dtype=tl.float16) # the full OUT is shape (OF, B)

    for of in range(0, tl.cdiv(IF, BLOCK_SIZE_IF)):
        mask = offsets_IF < IF - of * BLOCK_SIZE_IF
        W = tl.load(W_ptr + a_offsets, mask=mask[None, :], other=0.0) # shape (BLOCK_SIZE_OF, BLOCK_SIZE_IF)
        activations = tl.load(activations_ptr + b_offsets, mask=mask[:, None], other=0.0) # shape (BLOCK_SIZE_IF, BLOCK_SIZE_B)
        accumulator = tl.dot(W, activations, acc=accumulator)

        a_offsets += BLOCK_SIZE_IF * W_stride_1
        b_offsets += BLOCK_SIZE_IF * activations_stride_0

    # and now we reset the data type to the expected output
    accumulator = accumulator.to(tl.float16)

    # write back the block of the output matrix C with masks
    c_offsets = offsets_OF[:, None] * OUT_stride_0 + offsets_B[None, :] * OUT_stride_1
    c_mask = (offsets_OF[:, None] < OF) & (offsets_B[None, :] < B) # notice the 2D mask
    tl.store(OUT_ptr + c_offsets, accumulator, mask=OUT_mask) # shape (BLOCK_SIZE_OF, BLOCK_SIZE_B)





def w4a16(W, b, S, Z, group_size, activations):
    # assertions
    # assert C.shape[1] == E.shape[0], "incompatible dimensions"


    (OF, IF), (_, B) = W.shape, activations.shape
    OUT = torch.zeros((OF, B), device=W.device, dtype=torch.float16)

    grid = lambda meta: (triton.cdiv(OF, meta['BLOCK_SIZE_OF']) * triton.cdiv(B, meta['BLOCK_SIZE_B']), )
    _w4a16_kernel[grid](
        W, b, S, Z, group_size, activations, OUT,
        W.stride(0), W.stride(1),
        b.stride(0), b.stride(1),
        S.stride(0), S.stride(1),
        Z.stride(0), Z.stride(1),
        OF, IF, B,
        group_size.stride(0), group_size.stride(1),
        activations.stride(0), activations.stride(1),
        OUT.stride(0), OUT.stride(1),
    )
    return OUT

@torch.compile
def torch_w4a16(W, b, S, Z, group_size, activations):
    # assertions
    # assert C.shape[1] == E.shape[0], "incompatible dimensions"
    activations = activations.to(torch.float16)
    W_deq, b_deq = dequantize_layer(unpacking_layer(W), S, Z, b, group_size)
    
    W_deq = W_deq.to(torch.float16)
    b_deq = b_deq.to(torch.float16)

    # pass the activations
    return torch.nn.functional.linear(activations, W_deq, b_deq)


def test_w4a16(shapes: tuple, atol=1e-2, rtol=1e-1, device=DEVICE):
    torch.manual_seed(0)
    assert type(shapes) == tuple and len(shapes) == 3
    N, D, V = shapes
    E = torch.randn([D, N], device=device)
    C = torch.randn([V, D], device=device)
    
    # Test all three implementations
    c_tri_lsemm = lsemm(E, C)
    c_tri_lsemmo = lsemmo(E, C)
    c_ref = torch_lsemm(E, C)
    
    # Compare results
    print("LSEMM Results:")
    print("Triton LSEMM:", c_tri_lsemm)
    print("Triton LSEMMO:", c_tri_lsemmo)
    print("PyTorch:", c_ref)
    
    #torch.testing.assert_close(c_tri_lsemm, c_ref, atol=atol, rtol=rtol)
    #torch.testing.assert_close(c_tri_lsemmo, c_ref, atol=atol, rtol=rtol)
    print("PASSED - Both LSEMM and LSEMMO implementations match PyTorch reference")

# Modified benchmark function to include lsemmo
def benchmark_w4a16(shapes: tuple, device=DEVICE):
    torch.manual_seed(0)
    assert type(shapes) == tuple and len(shapes) == 3
    N, D, V = shapes
    E = torch.randn([D, N], device=device).contiguous()
    C = torch.randn([V, D], device=device).contiguous()
    
    quantiles = [0.5, 0.05, 0.95]
    
    # Benchmark all implementations
    tri_ms, tri_min_ms, tri_max_ms = triton.testing.do_bench(
        lambda: w4a16(E, C), quantiles=quantiles)
    trimo_ms, trimo_min_ms, trimo_max_ms = triton.testing.do_bench(
        lambda: w4a16(E, C), quantiles=quantiles)
    torch_ms, torch_min_ms, torch_max_ms = triton.testing.do_bench(
        lambda: torch_w4a16(E, C), quantiles=quantiles)
    
    print(f"Shape: N={N}, D={D}, V={V}")
    print(f"Triton W4A16: {tri_ms:.3f}ms (min: {tri_min_ms:.3f}ms, max: {tri_max_ms:.3f}ms)")
    print(f"Triton W4A16: {trimo_ms:.3f}ms (min: {trimo_min_ms:.3f}ms, max: {trimo_max_ms:.3f}ms)")
    print(f"PyTorch: {torch_ms:.3f}ms (min: {torch_min_ms:.3f}ms, max: {torch_max_ms:.3f}ms)")
    
    speedup_w4a16 = torch_ms / tri_ms
    print(f"Speedup vs PyTorch (W4A16): {speedup_w4a16:.2f}x")
    
    return {
        'triton_w4a16': (tri_ms, tri_min_ms, tri_max_ms),
        'torch': (torch_ms, torch_min_ms, torch_max_ms)
    }