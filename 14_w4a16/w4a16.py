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
    GROUP_SIZE: tl.constexpr,  num_stages: tl.constexpr,
):
    raw_PID_OF = tl.program_id(axis=0)
    raw_PID_B = tl.program_id(axis=1)
    
    num_PID_along_OF = tl.cdiv(OF, BLOCK_SIZE_OF)
    num_PID_along_B = tl.cdiv(B, BLOCK_SIZE_B)
    
    PID_half = raw_PID_OF // num_PID_along_OF 
    logical_PID_OF = raw_PID_OF % num_PID_along_OF
    
    PID = raw_PID_B * num_PID_along_OF + logical_PID_OF
    
    # Group-major ordering
    num_PID_in_group = GROUP_SIZE * num_PID_along_B
    group_id = PID // num_PID_in_group 
    first_PID_in_group_along_OF = group_id * GROUP_SIZE 
    group_size_adj = min(num_PID_along_OF - first_PID_in_group_along_OF, GROUP_SIZE) 
    
    PID_OF = first_PID_in_group_along_OF + ((PID % num_PID_in_group) % group_size_adj)
    PID_B = (PID % num_PID_in_group) // group_size_adj

    offsets_OF = PID_OF * BLOCK_SIZE_OF + tl.arange(0, BLOCK_SIZE_OF)

    offsets_B = PID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)

    offsets_IF = tl.arange(0, BLOCK_SIZE_IF)

    offsets_OF_scales = offsets_OF[:, None] * S_stride_0 + offsets_IF[None, :] * S_stride_1
    
    a_offsets = offsets_OF[:, None] * W_stride_0 + offsets_IF[None, :] * W_stride_1 # (OF block, IF block)
    b_offsets = offsets_IF[:, None] * activations_stride_0 + offsets_B[None, :] * activations_stride_1 # (IF block, B block)

    # did not group over this
    offsets_OF_G_adj = offsets_OF
    # repeating pointers because some are the same
    offsets_IF_G_adj = offsets_IF // group_size

    s_offsets = offsets_OF_G_adj[:, None] * S_stride_0 + offsets_IF_G_adj[None, :] * S_stride_1
    z_offsets = offsets_OF_G_adj[:, None] * Z_stride_0 + offsets_IF_G_adj[None, :] * Z_stride_1

    sz_mask_0 = offsets_OF_G_adj < OFG
    # inputs tensors are fp16 but we accumulate into a block of fp32 values for higher accuracy (we'll revert later)
    accumulator = tl.zeros((BLOCK_SIZE_OF, BLOCK_SIZE_B), dtype=tl.float32) # the full OUT is shape (OF, B)

    for step in range(0, tl.cdiv(IF, BLOCK_SIZE_IF)):\
        # repeating pointers because some are the same
        offsets_IF_G_adj = offsets_IF // group_size
        # Mask for OF and IF
        mask = offsets_IF < IF - step * BLOCK_SIZE_IF

        w_mask = (offsets_OF[:, None] < OF) & mask[None, :]
        act_mask = mask[:, None] & (offsets_B[None, :] < B)

        sz_mask_1 = offsets_IF_G_adj < IF - step * BLOCK_SIZE_IF
        sz_mask = sz_mask_0[:, None] & sz_mask_1[None, :]

        S = tl.load(S_ptr + s_offsets, mask=sz_mask, other=0.0) # shape (BLOCK_SIZE_OF, BLOCK_SIZE_IF)
        Z = tl.load(Z_ptr + z_offsets, mask=sz_mask, other=0.0) # shape (BLOCK_SIZE_OF, BLOCK_SIZE_IF)
        W = tl.load(W_ptr + a_offsets, mask=mask[None, :], other=0.0) # shape (BLOCK_SIZE_OF, BLOCK_SIZE_IF)
        activations = tl.load(activations_ptr + b_offsets, mask=mask[:, None], other=0.0) # shape (BLOCK_SIZE_IF, BLOCK_SIZE_B)

        # notice the weights are packed 8-bit values, so we need to unpack them and dequantize them
        W_h = tl.where(PID_half == 0, W & 0x0F, (W >> 4) & 0x0F) # Saving branch
        W_h = W_h.to(tl.float16)
        W_h -= Z
        W_h *= S

        accumulator = tl.dot(W_h, activations, acc=accumulator)

        a_offsets += BLOCK_SIZE_IF * W_stride_1
        b_offsets += BLOCK_SIZE_IF * activations_stride_0
        s_offsets += BLOCK_SIZE_IF * S_stride_1
        z_offsets += BLOCK_SIZE_IF * Z_stride_1
        offsets_IF += BLOCK_SIZE_IF


    # shifting output offsets to consider different halves
    offsets_OF = offsets_OF + (PID_half * num_PID_along_OF * BLOCK_SIZE_OF)

    # add bias per output feature and then reset the data type to the expected output
    b_offsets = offsets_OF * b_stride_0
    b_mask = offsets_OF < OF
    b_vals = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)
    accumulator += b_vals[:, None]
    accumulator = accumulator.to(tl.float16)

    # write back the block of the output matrix C with masks
    # Add this after calculating offsets_OF
    c_offsets = offsets_OF[:, None] * OUT_stride_0 + offsets_B[None, :] * OUT_stride_1
    c_mask = (offsets_OF[:, None] < OF) & (offsets_B[None, :] < B) # notice the 2D mask
    tl.store(OUT_ptr + c_offsets, accumulator, mask=c_mask) # shape (BLOCK_SIZE_OF, BLOCK_SIZE_B)


def w4a16(W, b, S, Z, group_size, activations):
    # assertions
    assert W.shape[1] == activations.shape[0], "incompatible dimensions"

    (OF, IF), (_, B), (_, OFG) = W.shape, activations.shape, S.shape

    OUT = torch.zeros((2*OF, B), device=W.device, dtype=torch.float16)

    grid = lambda meta: ((2 * triton.cdiv(OF, meta['BLOCK_SIZE_OF'])), triton.cdiv(B, meta['BLOCK_SIZE_B']))
    _w4a16_kernel[grid](
        W, b, S, Z, activations, OUT,
        W.stride(0), W.stride(1),
        b.stride(0), b.stride(1),
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
    """
    # Extract lower 4 bits
    w0 = W_qp & 0x0F
    # Extract upper 4 bits and shift right
    w1 = (W_qp >> 4) & 0x0F

    # Concatenate back along the K dimension
    wF = torch.cat([w0, w1], dim=-1)
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
    # assertions
    # assert C.shape[1] == E.shape[0], "incompatible dimensions"
    activations = activations.to(torch.float16)
    W_deq, b_deq = dequantize_layer(unpacking_layer(W), S, Z, b, group_size)
    
    W_deq = W_deq.to(torch.float16)
    b_deq = b_deq.to(torch.float16)

    # pass the activations
    return torch.nn.functional.linear(activations, W_deq, b_deq)

def test_w4a16(shapes: tuple, atol=1e-2, rtol=1e-1, device=DEVICE):
    """
    Unit test for the Triton w4a16 kernel.

    `shapes` is a 3-tuple `(OF, IF, B)` where:
      - OF: number of output features
      - IF: number of input features
      - B:  batch size

    This test constructs a special packed-weight tensor so that, after the
    unpacking logic in the kernel, the effective weights are simply the
    low 4 bits of each entry. It then checks that the Triton kernel matches a
    straightforward PyTorch matmul + bias using those effective weights.
    """
    import math

    torch.manual_seed(0)
    assert isinstance(shapes, tuple) and len(shapes) == 3
    OF, IF, B = shapes

    # Choose a fixed group size; it only affects how S/Z are indexed inside the
    # kernel, not the reference computation here.
    group_size = 16

    # Construct packed 8-bit weights where high and low nibbles are identical.
    # That way, regardless of which nibble the kernel selects, the effective
    # 4-bit value is the same.
    v = torch.randint(0, 16, (OF, IF), device=device, dtype=torch.int16)  # 4-bit values
    W_packed = (v | (v << 4)).to(torch.int16)  # [0, 255], same nibble in low/high

    # Bias per output feature
    b = torch.randn((OF,), device=device, dtype=torch.float16)

    # Scales and zero-points: fill with ones/zeros, but make the tensor large
    # enough so all kernel index math stays in-bounds and masks are always valid.
    L = max(64, IF, math.ceil(OF / group_size))
    S = torch.ones((L, L), device=device, dtype=torch.float16)
    Z = torch.zeros_like(S)

    # Activations: (IF, B)
    activations = torch.randn((IF, B), device=device, dtype=torch.float16)

    # Triton kernel output
    out_triton = w4a16(W_packed, b, S, Z, group_size, activations)

    # Reference weights: just the low nibble interpreted as FP16
    W_ref = (W_packed & 0x0F).to(torch.float16)
    out_ref = torch.matmul(W_ref, activations) + b[:, None]

    torch.testing.assert_close(out_triton, out_ref, atol=atol, rtol=rtol)
    print("w4a16 unit test PASSED")


def benchmark_w4a16(shapes: tuple, device=DEVICE):
    """
    Benchmark the Triton w4a16 kernel against a PyTorch matmul + bias on the
    same effective weights.

    `shapes` is a 3-tuple `(OF, IF, B)` as in `test_w4a16`.
    """
    import math

    torch.manual_seed(0)
    assert isinstance(shapes, tuple) and len(shapes) == 3
    OF, IF, B = shapes

    group_size = 16

    # Packed weights with identical nibbles (see test_w4a16 for rationale)
    v = torch.randint(0, 16, (OF, IF), device=device, dtype=torch.int16)
    W_packed = (v | (v << 4)).to(torch.int16).contiguous()

    b = torch.randn((OF,), device=device, dtype=torch.float16).contiguous()

    L = max(64, IF, math.ceil(OF / group_size))
    S = torch.ones((L, L), device=device, dtype=torch.float16).contiguous()
    Z = torch.zeros_like(S).contiguous()

    activations = torch.randn((IF, B), device=device, dtype=torch.float16).contiguous()

    # Effective reference weights
    W_ref = (W_packed & 0x0F).to(torch.float16).contiguous()

    quantiles = [0.5, 0.05, 0.95]

    # Triton kernel benchmark
    tri_ms, tri_min_ms, tri_max_ms = triton.testing.do_bench(
        lambda: w4a16(W_packed, b, S, Z, group_size, activations),
        quantiles=quantiles,
    )

    # PyTorch reference benchmark: matmul + bias
    def torch_ref():
        return torch.matmul(W_ref, activations) + b[:, None]

    torch_ms, torch_min_ms, torch_max_ms = triton.testing.do_bench(
        torch_ref,
        quantiles=quantiles,
    )

    print(f"Shape: OF={OF}, IF={IF}, B={B}")
    print(f"Triton W4A16: {tri_ms:.3f}ms (min: {tri_min_ms:.3f}ms, max: {tri_max_ms:.3f}ms)")
    print(f"PyTorch matmul+bias: {torch_ms:.3f}ms (min: {torch_min_ms:.3f}ms, max: {torch_max_ms:.3f}ms)")

    speedup_w4a16 = torch_ms / tri_ms
    print(f"Speedup vs PyTorch (W4A16): {speedup_w4a16:.2f}x")

    return {
        'triton_w4a16': (tri_ms, tri_min_ms, tri_max_ms),
        'torch_matmul_bias': (torch_ms, torch_min_ms, torch_max_ms),
    }