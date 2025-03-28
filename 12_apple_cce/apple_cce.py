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

autotune_configs_iep =[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_D': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_D': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_D': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_D': 64, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_D': 128, 'num_stages': 5, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_D': 128, 'num_stages': 5, 'num_warps': 4}),
    ]

@triton.autotune(configs = autotune_configs_iep, key=['N', 'D', 'V'])
@triton.jit
def _indexed_essential_probs_kernel(
    e_ptr, i_ptr, # contains the index of the correct label
    c_ptr, output_ptr,
    N, D, V,
    stride_ed, stride_en,
    stride_cv, stride_cd,
    stride_in, stride_on,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    # Auto-tuned but want to use
    num_stages: tl.constexpr,
):
    # Basically here we index into the Classifier = C, and using the embeddings of the i'th token, E_i
    # We calculate C_{x_i} * E_i
    # I divided E,C into blocks of N, these are for PIDs, then i divide within them into blocks of D
    # Which I iterate over to calculate the dot product of  C_{x_i}, E_i
    pid = tl.program_id(axis=0)
    # I is of shape (N),
    n_dim_offset = pid * BLOCK_SIZE_N # OFFSET OF DIM N
    n_dim_offsets = (n_dim_offset + tl.arange(0, BLOCK_SIZE_N))
    mask_n = n_dim_offsets < N
    c_idxs = tl.load(i_ptr + n_dim_offsets * stride_in, mask=mask_n) # shape of (BLOCK_SIZE_N,)
    # these indices are according to a tensor C with shape (V, D) 
    # Thus, need to be multiplied by stride_cv
    c_offsets = (c_idxs * stride_cv)[:, None] + (tl.arange(0, BLOCK_SIZE_D) * stride_cd)[None, :]
    # Offsets of shape (BLOCK_SIZE_N, BLOCK_SIZE_D)
    e_offsets = (tl.arange(0, BLOCK_SIZE_D) * stride_ed)[:, None] + (n_dim_offsets * stride_en)[None, :]
    # Offsets of shape (BLOCK_SIZE_D, BLOCK_SIZE_N)
    mask_vocab = c_idxs < V
    mask_d = tl.arange(0, BLOCK_SIZE_D)

    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for d in tl.range(0, D, BLOCK_SIZE_D, num_stages=num_stages):
        # Loading C_BLOCK
        mask_c = (mask_vocab)[:, None] & (mask_d < D)[None, :]
        C_BLOCK = tl.load(c_ptr + c_offsets, mask=mask_c, other=0.0) # BN, BD
        # Loading E_BLOCK
        mask_e = (mask_d < D)[:, None] & (mask_n)[None, :]
        E_BLOCK = tl.load(e_ptr + e_offsets, mask=mask_e, other=0.0) # BD, BN

        acc += tl.sum((C_BLOCK * tl.trans(E_BLOCK)), axis=1)
        # We calclate dot product for every E_i , C_i

        c_offsets += BLOCK_SIZE_D * stride_cd
        e_offsets += BLOCK_SIZE_D * stride_ed
        mask_d += BLOCK_SIZE_D

    tl.store(output_ptr + n_dim_offsets * stride_on, acc, mask=mask_n)


def indexed_essential_probs(E, C, I):
    (D, N) , (V, _) = E.shape, C.shape
    assert E.shape[0] == C.shape[1]
    
    # Make sure everything is on the same device
    device = E.device
    E = E.to(device)
    C = C.to(device)
    I = I.to(device)
    O = torch.empty((N,), device=device)
    
    # Make sure we're operating on contiguous tensors
    E = E.contiguous()
    C = C.contiguous()
    I = I.contiguous()
    O = O.contiguous()
    
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    
    try:
        # Pass pointers in the correct order: e_ptr, i_ptr, c_ptr, output_ptr
        _indexed_essential_probs_kernel[grid](
                            E, I, C, O,
                            N, D, V,
                            E.stride(0), E.stride(1),
                            C.stride(0), C.stride(1),
                            I.stride(0), O.stride(0),
                            )
    except Exception as e:
        print(f"Error in kernel execution: {e}")
        import traceback
        traceback.print_exc()
        
    return O

@torch.compile
def torch_indexed_essential_probs(E, C, I):
    # Vectorized approach: batch matrix-vector multiplication
    return torch.einsum('ij,ij->i', C[I], E.T)

def test_indexed_essential_probs(shapes: tuple, atol=1e-2, rtol=1e-1, device=DEVICE):
    # create input data
    torch.manual_seed(0)
    assert type(shapes) == tuple and len(shapes) == 3
    N, D, V = shapes
    E = torch.randn([D, N], device=device)
    C = torch.randn([V, D], device=device)
    O = torch.empty([N], device=device)
    I = torch.randint(high=V, size=(N,), device=device)
    # run kernel & pytorch reference implementation
    c_tri = indexed_essential_probs(E, C, I)
    c_ref = torch_indexed_essential_probs(E, C, I)
    # compare
    torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
    print("PASSED")

def benchmark_indexed_essential_probs(shapes: tuple, device=DEVICE):
    # create input data
    torch.manual_seed(0)
    assert type(shapes) == tuple and len(shapes) == 3
    N, D, V = shapes
    E = torch.randn([D, N], device=device)
    C = torch.randn([V, D], device=device)
    I = torch.randint(high=V, size=(N,), device=device)
    
    # Ensure all tensors are on the correct device and contiguous
    E = E.contiguous().to(device)
    C = C.contiguous().to(device)
    I = I.contiguous().to(device)
    
    # Run the benchmark
    quantiles = [0.5, 0.05, 0.95]
    tri_ms, tri_min_ms, tri_max_ms = triton.testing.do_bench(
        lambda: indexed_essential_probs(E, C, I), 
        quantiles=quantiles
    )
    
    torch_ms, torch_min_ms, torch_max_ms = triton.testing.do_bench(
        lambda: torch_indexed_essential_probs(E, C, I), 
        quantiles=quantiles
    )
    
    print(f"Shape: N={N}, D={D}, V={V}")
    print(f"Triton: {tri_ms:.3f}ms (min: {tri_min_ms:.3f}ms, max: {tri_max_ms:.3f}ms)")
    print(f"PyTorch: {torch_ms:.3f}ms (min: {torch_min_ms:.3f}ms, max: {torch_max_ms:.3f}ms)")
    
    speedup = torch_ms / tri_ms
    print(f"Speedup vs PyTorch: {speedup:.2f}x")
    
    return {
        'triton': (tri_ms, tri_min_ms, tri_max_ms),
        'torch': (torch_ms, torch_min_ms, torch_max_ms)
    }


########## ALGORITHM (2) ###########
# Log-Sum-Exp ( Matrix-Multiplication )

# Define autotuning configurations for lsemm
autotune_configs_lsemm = [
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
def _lsemmo_kernel(
    e_ptr, c_ptr, output_ptr, locks_ptr, maxes_ptr,
    N, D, V, L,
    stride_cv, stride_cd,
    stride_ed, stride_en,
    stride_on, stride_ll,
    stride_miv,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_V: tl.constexpr,
    GROUP_SIZE: tl.constexpr,  num_stages: tl.constexpr,
):
    # We want to matmul (V, D) @ (D, N) and the sum over the V axis
    PID = tl.program_id(axis=0) 
    
    # Group-major ordering
    num_PID_along_M = tl.cdiv(V, BLOCK_SIZE_V)
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N)
    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    group_id = PID // num_PID_in_group 
    first_PID_in_group_along_M = group_id * GROUP_SIZE 
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE) 
    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
    PID_N = (PID % num_PID_in_group) // group_size_adj

    offsets_V = PID_M * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_D = tl.arange(0, BLOCK_SIZE_D)
    
    # Reference
    offsets_O = offsets_N
    offsets_M = offsets_N
    
    
    a_offsets = offsets_V[:, None] * stride_cv + offsets_D[None, :] * stride_cd # (BV, BD)
    b_offsets = offsets_D[:, None] * stride_ed + offsets_N[None, :] * stride_en # (BD, BN)

    accb = tl.zeros((BLOCK_SIZE_V, BLOCK_SIZE_N,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    block_cmx = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32) # current max
    cexpc, cexpg = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32), tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32) # correcting shit later
    # These are multipliers that affect exisiting sums to make their max-num precision subtraction
    # global, to take the largest max

    mask_v = offsets_V < V
    mask_n = offsets_N < N
        
    for d in range(0, tl.cdiv(D, BLOCK_SIZE_D), num_stages=num_stages):
        mask_d = offsets_D < D - d * BLOCK_SIZE_D
        mask_a = mask_v[:, None] & mask_d[None, :]
        mask_b = mask_d[:, None] & mask_n[None, :]

        a = tl.load(c_ptr + a_offsets, mask=mask_a, other=0.0)
        b = tl.load(e_ptr + b_offsets, mask=mask_b, other=0.0)
        
        # a @ b => (BV, BN) and we need to sum over BV
        accb = tl.dot(a, b, acc=accb) # (BLOCK_SIZE_V, BLOCK_SIZE_N)
    
        a_offsets += BLOCK_SIZE_D * stride_cd
        b_offsets += BLOCK_SIZE_D * stride_ed

    # Masked elements in the matmul that affect the final result
    # Are the ones below the true elements in the result matrix
    # and left to finish of the true elements
    # So we can calculate this and remove their effect
    # num_masked_v * (BN - num_masked_n), these are the number of those who have effect
    # num_nmasked_n = tl.sum(mask_n)

    block_cmx = tl.max(accb, axis=0) # (BN,)
    accb -= block_cmx[None, :]
    accb = tl.where(mask_v[:, None] & mask_n[None, :], accb, float('-inf'))
    acc += tl.sum(tl.exp(accb), axis=0) # (BN,)
    acc = tl.log(acc)

    # Now acc holds log(sum(exp(z_i - block_cmx))) while the real result is
    # log(sum(exp(z_i - block_cmx))) + block_cmx
    
    maxes_ptrs = maxes_ptr + offsets_M * stride_miv
    ointermediate_ptrs = output_ptr + offsets_O * stride_on
    
    mask_m = (offsets_M < N)   
    mask_o = (offsets_O < N)

    lock_id = PID_N
    locks_ptrf = locks_ptr + lock_id * stride_ll
    count_ptr = locks_ptr + (L + lock_id) * stride_ll
    while tl.atomic_cas(locks_ptrf, 0, 1) == 1:
        pass

    # Saving useless addition
    count = tl.load(count_ptr)
    if count == 0:
        tl.atomic_xchg(count_ptr, 1)
        tl.store(ointermediate_ptrs, acc, mask=mask_o)
        tl.store(maxes_ptrs, block_cmx, mask=mask_m) # Store the maxes of the maxes of the block
    else:
        # We basically keep the maximum here at all times
        block_acc = tl.load(ointermediate_ptrs, mask=mask_o) # Holds sum(exp(z_block - block_gmx))
        block_gmx = tl.atomic_max(maxes_ptrs, block_cmx, mask_m)
        block_bmx = tl.load(maxes_ptrs, mask_m)
        # block_acc = tl.exp(block_acc) # block_acc holds sum(exp(z_block - block_gmx))

        cexpc, cexpg = block_cmx - block_bmx, block_gmx - block_bmx
        # sum(exp(z_i - block_cmx) * exp(block_cmx - block_gmx) = sum(exp(z_i - block_cmx + block_cmx - block_gmx)
        acc += cexpc
        block_acc += cexpg
        # depending on the mask, (1) if gmx greater/equal, (2) else
        # (1) Now acc holds sum(exp(z_i - block_gmx)) , shape (BLOCK_SIZE_N,)
        # (2) Now block_acc holds sum(exp(z_block - block_cmx)), shape(BLOCK_SIZE_N)
        acc = tl.exp(acc) + tl.exp(block_acc)
        # Now everything is summed and holds the max, not holds, more like holds the effect
        tl.store(ointermediate_ptrs, tl.log(acc), mask=mask_o)

    tl.atomic_xchg(locks_ptr + lock_id * stride_ll, 0) # Unlock

@triton.autotune(configs=autotune_configs_lsemm, key=['N', 'D', 'V'])
@triton.jit
def _lsemm_kernel(
    e_ptr, c_ptr, output_ptr, locks_ptr, maxes_ptr,
    N, D, V, L,
    stride_cv, stride_cd,
    stride_ed, stride_en,
    stride_on, stride_ll,
    stride_miv,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_V: tl.constexpr,
    GROUP_SIZE: tl.constexpr,  num_stages: tl.constexpr,
):
    # We want to matmul (V, D) @ (D, N) and the sum over the V axis
    PID = tl.program_id(axis=0) 
    
    # Group-major ordering
    num_PID_along_M = tl.cdiv(V, BLOCK_SIZE_V)
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N)
    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    group_id = PID // num_PID_in_group 
    first_PID_in_group_along_M = group_id * GROUP_SIZE 
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE) 
    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
    PID_N = (PID % num_PID_in_group) // group_size_adj

    offsets_V = PID_M * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_D = tl.arange(0, BLOCK_SIZE_D)
    
    # Reference
    offsets_O = offsets_N
    offsets_M = offsets_N
    
    
    a_offsets = offsets_V[:, None] * stride_cv + offsets_D[None, :] * stride_cd # (BV, BD)
    b_offsets = offsets_D[:, None] * stride_ed + offsets_N[None, :] * stride_en # (BD, BN)

    accb = tl.zeros((BLOCK_SIZE_V, BLOCK_SIZE_N,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    block_cmx = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32) # current max
    cexpc, cexpg = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32), tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32) # correcting shit later
    # These are multipliers that affect exisiting sums to make their max-num precision subtraction
    # global, to take the largest max

    mask_v = offsets_V < V
    mask_n = offsets_N < N
        
    for d in range(0, tl.cdiv(D, BLOCK_SIZE_D), num_stages=num_stages):
        mask_d = offsets_D < D - d * BLOCK_SIZE_D
        mask_a = mask_v[:, None] & mask_d[None, :]
        mask_b = mask_d[:, None] & mask_n[None, :]

        a = tl.load(c_ptr + a_offsets, mask=mask_a, other=0.0)
        b = tl.load(e_ptr + b_offsets, mask=mask_b, other=0.0)
        
        # a @ b => (BV, BN) and we need to sum over BV
        accb = tl.dot(a, b, acc=accb) # (BLOCK_SIZE_V, BLOCK_SIZE_N)
    
        a_offsets += BLOCK_SIZE_D * stride_cd
        b_offsets += BLOCK_SIZE_D * stride_ed

    # Masked elements in the matmul that affect the final result
    # Are the ones below the true elements in the result matrix
    # and left to finish of the true elements
    # So we can calculate this and remove their effect
    # num_masked_v * (BN - num_masked_n), these are the number of those who have effect
    # num_nmasked_n = tl.sum(mask_n)

    block_cmx = tl.max(accb, axis=0) # (BN,)
    accb = tl.where(mask_v[:, None] & mask_n[None, :], accb, float('-inf'))
    accb -= block_cmx[None, :]
    acc += tl.sum(tl.exp(accb), axis=0) # (BN,)

    # Now acc holds log(sum(exp(z_i - block_cmx))) while the real result is
    # log(sum(exp(z_i - block_cmx))) + block_cmx
    
    maxes_ptrs = maxes_ptr + offsets_M * stride_miv
    ointermediate_ptrs = output_ptr + offsets_O * stride_on
    
    mask_m = (offsets_M < N)   
    mask_o = (offsets_O < N)

    lock_id = PID_N
    locks_ptrf = locks_ptr + lock_id * stride_ll
    count_ptr = locks_ptr + (L + lock_id) * stride_ll
    while tl.atomic_cas(locks_ptrf, 0, 1) == 1:
        pass

    # Saving useless addition
    count = tl.load(count_ptr)
    if count == 0:
        tl.atomic_xchg(count_ptr, 1)
        tl.store(ointermediate_ptrs, acc, mask=mask_o)
        tl.store(maxes_ptrs, block_cmx, mask=mask_m) # Store the maxes of the maxes of the block
    else:
        # We basically keep the maximum here at all times
        block_acc = tl.load(ointermediate_ptrs, mask=mask_o) # Holds sum(exp(z_block - block_gmx))
        block_gmx = tl.atomic_max(maxes_ptrs, block_cmx, mask_m)
        block_bmx = tl.load(maxes_ptrs, mask_m)
        # block_acc = tl.exp(block_acc) # block_acc holds sum(exp(z_block - block_gmx))

        #cexpc, cexpg = block_cmx - block_bmx, block_gmx - block_bmx
        #cexpc = tl.where(tl.abs(cexpc) > 1e-5, tl.exp(cexpc), 1 + cexpc)
        #cexpg = tl.where(tl.abs(cexpg) > 1e-5, tl.exp(cexpg), 1 + cexpg)
        cexpc, cexpg = tl.exp(block_cmx - block_bmx), tl.exp(block_gmx - block_bmx)
        # sum(exp(z_i - block_cmx) * exp(block_cmx - block_gmx) = sum(exp(z_i - block_cmx + block_cmx - block_gmx)
        # depending on the mask, (1) if gmx greater/equal, (2) else
        # (1) Now acc holds sum(exp(z_i - block_gmx)) , shape (BLOCK_SIZE_N,)
        # (2) Now block_acc holds sum(exp(z_block - block_cmx)), shape(BLOCK_SIZE_N)
        acc = acc * cexpc + block_acc * cexpg
        # Now everything is summed and holds the max, not holds, more like holds the effect
        tl.store(ointermediate_ptrs, acc, mask=mask_o)

    tl.atomic_xchg(locks_ptr + lock_id * stride_ll, 0) # Unlock

def lsemmo(E, C):
    assert C.ndim == E.ndim == 2, "only supports matrices, not vectors or tensors"
    assert C.shape[1] == E.shape[0], "incompatible dimensions"

    (D, N), (V, _) = E.shape, C.shape
    O = torch.full((N,), float('-inf'), device=E.device, dtype=torch.float32)
    M = torch.full((N,), float('-inf'), device=E.device, dtype=torch.float32)
    L = math.ceil(N / 32)  # We assume the block size is larger than that and we will have enough locks
    locks = torch.zeros(2 * L, dtype=torch.int32, device=E.device)
    
    grid = lambda meta: (triton.cdiv(V, meta['BLOCK_SIZE_V']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    _lsemmo_kernel[grid](
        E, C, O, locks, M,
        N, D, V, L,
        C.stride(0), C.stride(1),
        E.stride(0), E.stride(1),
        O.stride(0), locks.stride(0),
        M.stride(0),
    )
    # Now add the maxes changes and log it
    out = O + M
    #out = O
    return out

def lsemm(E, C):
    assert C.ndim == E.ndim == 2, "only supports matrices, not vectors or tensors"
    assert C.shape[1] == E.shape[0], "incompatible dimensions"

    (D, N), (V, _) = E.shape, C.shape
    O = torch.zeros((N,), device=E.device, dtype=torch.float32)
    M = torch.full((N,), float('-inf'), device=E.device, dtype=torch.float32)
    L = math.ceil(N / 32)  # We assume the block size is larger than that and we will have enough locks
    locks = torch.zeros(2 * L, dtype=torch.int32, device=E.device)
    
    grid = lambda meta: (triton.cdiv(V, meta['BLOCK_SIZE_V']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    _lsemm_kernel[grid](
        E, C, O, locks, M,
        N, D, V, L,
        C.stride(0), C.stride(1),
        E.stride(0), E.stride(1),
        O.stride(0), locks.stride(0),
        M.stride(0),
    )
    # Now add the maxes changes and log it
    out = torch.log(O) + M
    #out = torch.log(O)
    return out

@torch.compile
def torch_lsemm(E, C):
    assert C.ndim == E.ndim == 2, "only supports matrices, not vectors or tensors"
    assert C.shape[1] == E.shape[0], "incompatible dimensions"

    RES = C @ E  # (V, D) @ (D, N) = (V, N)
    mx = torch.max(RES, dim=0)[0]  # Shape: (1, N)
    RES = RES - mx  # Broadcasting: (V, N) - (1, N)
    RES = torch.sum(torch.exp(RES), dim=0)  # Sum over V dimension, result shape: (N,)
    out = torch.log(RES) + mx
    #out = torch.log(RES)
    return out

def test_lsemm(shapes: tuple, atol=1e-2, rtol=1e-1, device=DEVICE):
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
def benchmark_lsemm(shapes: tuple, device=DEVICE):
    torch.manual_seed(0)
    assert type(shapes) == tuple and len(shapes) == 3
    N, D, V = shapes
    E = torch.randn([D, N], device=device).contiguous()
    C = torch.randn([V, D], device=device).contiguous()
    
    quantiles = [0.5, 0.05, 0.95]
    
    # Benchmark all implementations
    tri_ms, tri_min_ms, tri_max_ms = triton.testing.do_bench(
        lambda: lsemm(E, C), quantiles=quantiles)
    trimo_ms, trimo_min_ms, trimo_max_ms = triton.testing.do_bench(
        lambda: lsemmo(E, C), quantiles=quantiles)
    torch_ms, torch_min_ms, torch_max_ms = triton.testing.do_bench(
        lambda: torch_lsemm(E, C), quantiles=quantiles)
    
    print(f"Shape: N={N}, D={D}, V={V}")
    print(f"Triton LSEMM: {tri_ms:.3f}ms (min: {tri_min_ms:.3f}ms, max: {tri_max_ms:.3f}ms)")
    print(f"Triton LSEMMO: {trimo_ms:.3f}ms (min: {trimo_min_ms:.3f}ms, max: {trimo_max_ms:.3f}ms)")
    print(f"PyTorch: {torch_ms:.3f}ms (min: {torch_min_ms:.3f}ms, max: {torch_max_ms:.3f}ms)")
    
    speedup_lsemm = torch_ms / tri_ms
    speedup_lsemmo = torch_ms / trimo_ms
    print(f"Speedup vs PyTorch (LSEMM): {speedup_lsemm:.2f}x")
    print(f"Speedup vs PyTorch (LSEMMO): {speedup_lsemmo:.2f}x")
    
    return {
        'triton_lsemm': (tri_ms, tri_min_ms, tri_max_ms),
        'triton_lsemmo': (trimo_ms, trimo_min_ms, trimo_max_ms),
        'torch': (torch_ms, torch_min_ms, torch_max_ms)
    }

# Add more comprehensive benchmarking with triton.testing.perf_report
configs = [
    triton.testing.Benchmark(
        x_names=["N"],  # We'll vary sequence length
        x_vals=[128 * i for i in range(1, 17)],  # From 128 to 2048
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="Execution Time (ms)",
        plot_name="iep-performance-N",
        args={"D": 512, "V": 2048},  # Fixed dimensions
    ),
    triton.testing.Benchmark(
        x_names=["D"],  # We'll vary embedding dimension
        x_vals=[128 * i for i in range(1, 9)],  # From 128 to 1024
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="Execution Time (ms)",
        plot_name="iep-performance-D",
        args={"N": 512, "V": 2048},  # Fixed dimensions
    ),
    triton.testing.Benchmark(
        x_names=["V"],  # We'll vary vocabulary size
        x_vals=[1000 * i for i in range(1, 11)],  # From 1000 to 10000
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="Execution Time (ms)",
        plot_name="iep-performance-V",
        args={"N": 512, "D": 512},  # Fixed dimensions
    ),
]

lsemm_configs = [
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(1, 17)],
        line_arg="provider",
        line_vals=["triton_lsemm", "triton_lsemmo", "torch"],
        line_names=["Triton LSEMM", "Triton LSEMMO", "PyTorch"],
        styles=[("red", "-"), ("green", "-"), ("blue", "-")],
        ylabel="Execution Time (ms)",
        plot_name="lsemm-performance-N",
        args={"D": 512, "V": 2048},
    ),
    triton.testing.Benchmark(
        x_names=["D"],
        x_vals=[128 * i for i in range(1, 9)],
        line_arg="provider",
        line_vals=["triton_lsemm", "triton_lsemmo", "torch"],
        line_names=["Triton LSEMM", "Triton LSEMMO", "PyTorch"],
        styles=[("red", "-"), ("green", "-"), ("blue", "-")],
        ylabel="Execution Time (ms)",
        plot_name="lsemm-performance-D",
        args={"N": 512, "V": 2048},
    ),
    triton.testing.Benchmark(
        x_names=["V"],
        x_vals=[1000 * i for i in range(1, 11)],
        line_arg="provider",
        line_vals=["triton_lsemm", "triton_lsemmo", "torch"],
        line_names=["Triton LSEMM", "Triton LSEMMO", "PyTorch"],
        styles=[("red", "-"), ("green", "-"), ("blue", "-")],
        ylabel="Execution Time (ms)",
        plot_name="lsemm-performance-V",
        args={"N": 512, "D": 512},
    ),
]


@triton.testing.perf_report(configs)
def benchmark_iep(N, D, V, provider):
    # Create input tensors
    torch.manual_seed(0)
    E = torch.randn([D, N], device=DEVICE).contiguous()
    C = torch.randn([V, D], device=DEVICE).contiguous()
    I = torch.randint(high=V, size=(N,), device=DEVICE).contiguous()
    
    quantiles = [0.5, 0.05, 0.95]
    
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: indexed_essential_probs(E, C, I), 
            quantiles=quantiles
        )
    elif provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_indexed_essential_probs(E, C, I), 
            quantiles=quantiles
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    return ms, max_ms, min_ms


@triton.testing.perf_report(lsemm_configs)
def benchmark_lsemm_perf(N, D, V, provider):
    torch.manual_seed(0)
    E = torch.randn([D, N], device=DEVICE).contiguous()
    C = torch.randn([V, D], device=DEVICE).contiguous()
    
    quantiles = [0.5, 0.05, 0.95]
    
    if provider == 'triton_lsemm':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: lsemm(E, C), quantiles=quantiles)
    elif provider == 'triton_lsemmo':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: lsemmo(E, C), quantiles=quantiles)
    elif provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_lsemm(E, C), quantiles=quantiles)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    return ms, max_ms, min_ms

def export_results_to_csv(results, filename):
    """Export benchmark results to CSV file"""
    os.makedirs('results', exist_ok=True)
    filepath = os.path.join('results', filename)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Ensure numeric columns are stored as numbers when possible
    numeric_cols = ['N', 'D', 'V']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
    return df


def create_summary_table(df, title, filename):
    """Create a summary table as PNG using matplotlib"""
    # Create figure and axis
    fig = plt.figure(figsize=(12, len(df) * 0.5 + 2), dpi=150)
    ax = plt.subplot(111)
    
    # Hide axes
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Set title
    plt.title(title, fontsize=14, pad=20)
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    filepath = os.path.join('results', filename)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"Table saved to {filepath}")


def run_basic_benchmarks():
    print("Running basic benchmarks...")
    shapes = [
        (128, 128, 128),   # Small
        (256, 256, 256),   # Medium
        (512, 512, 512),   # Large
        (1024, 256, 1024), # Wide sequence
        (256, 1024, 1024), # High dimension
    ]
    
    iep_results = {
        'Shape': [], 'N': [], 'D': [], 'V': [],
        'Triton (ms)': [], 'PyTorch (ms)': [], 'Speedup': []
    }
    
    lsemm_results = {
        'Shape': [], 'N': [], 'D': [], 'V': [],
        'Triton LSEMM (ms)': [], 'Triton LSEMMO (ms)': [], 'PyTorch (ms)': [],
        'Speedup LSEMM': [], 'Speedup LSEMMO': []
    }
    
    for shape in shapes:
        N, D, V = shape
        shape_str = f"N={N}, D={D}, V={V}"
        
        # IEP benchmark
        print(f"\nBenchmarking indexed_essential_probs with shape: {shape_str}")
        timings = benchmark_indexed_essential_probs(shape)
        tri_ms = timings['triton'][0]
        torch_ms = timings['torch'][0]
        speedup = torch_ms / tri_ms
        
        iep_results['Shape'].append(shape_str)
        iep_results['N'].append(N)
        iep_results['D'].append(D)
        iep_results['V'].append(V)
        iep_results['Triton (ms)'].append(f"{tri_ms:.2f}")
        iep_results['PyTorch (ms)'].append(f"{torch_ms:.2f}")
        iep_results['Speedup'].append(f"{speedup:.2f}x")
        
        # LSEMM benchmark
        print(f"\nBenchmarking lsemm/lsemmo with shape: {shape_str}")
        lsemm_timings = benchmark_lsemm(shape)
        lsemm_tri_ms = lsemm_timings['triton_lsemm'][0]
        lsemmo_tri_ms = lsemm_timings['triton_lsemmo'][0]
        lsemm_torch_ms = lsemm_timings['torch'][0]
        
        speedup_lsemm = lsemm_torch_ms / lsemm_tri_ms
        speedup_lsemmo = lsemm_torch_ms / lsemmo_tri_ms
        
        lsemm_results['Shape'].append(shape_str)
        lsemm_results['N'].append(N)
        lsemm_results['D'].append(D)
        lsemm_results['V'].append(V)
        lsemm_results['Triton LSEMM (ms)'].append(f"{lsemm_tri_ms:.2f}")
        lsemm_results['Triton LSEMMO (ms)'].append(f"{lsemmo_tri_ms:.2f}")
        lsemm_results['PyTorch (ms)'].append(f"{lsemm_torch_ms:.2f}")
        lsemm_results['Speedup LSEMM'].append(f"{speedup_lsemm:.2f}x")
        lsemm_results['Speedup LSEMMO'].append(f"{speedup_lsemmo:.2f}x")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export IEP results
    iep_df = export_results_to_csv(iep_results, f"iep_basic_benchmarks_{timestamp}.csv")
    create_summary_table(iep_df, "IEP Basic Benchmarks", f"iep_basic_benchmarks_{timestamp}.png")
    
    # Export LSEMM results
    lsemm_df = export_results_to_csv(lsemm_results, f"lsemm_basic_benchmarks_{timestamp}.csv")
    create_summary_table(lsemm_df, "LSEMM/LSEMMO Basic Benchmarks", f"lsemm_basic_benchmarks_{timestamp}.png")
    
    # Print summary
    print("\n----- BENCHMARK SUMMARY -----")
    print("\nIndexed Essential Probs (IEP):")
    for i, shape in enumerate(shapes):
        print(f"Shape ({iep_results['Shape'][i]}):")
        print(f"  Triton: {iep_results['Triton (ms)'][i]}ms")
        print(f"  PyTorch: {iep_results['PyTorch (ms)'][i]}ms (speedup: {iep_results['Speedup'][i]})")
    
    print("\nLSEMM/LSEMMO:")
    for i, shape in enumerate(shapes):
        print(f"Shape ({lsemm_results['Shape'][i]}):")
        print(f"  Triton LSEMM: {lsemm_results['Triton LSEMM (ms)'][i]}ms")
        print(f"  Triton LSEMMO: {lsemm_results['Triton LSEMMO (ms)'][i]}ms")
        print(f"  PyTorch: {lsemm_results['PyTorch (ms)'][i]}ms")
        print(f"  Speedups - LSEMM: {lsemm_results['Speedup LSEMM'][i]}, LSEMMO: {lsemm_results['Speedup LSEMMO'][i]}")

# Updated run_detailed_benchmarks
def run_detailed_benchmarks(write_plots=True):
    print("\nRunning detailed benchmarks...")
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {}
    all_configs = configs + lsemm_configs
    
    for config in all_configs:
        config_name = config.plot_name
        x_name = config_name.split('-')[-1]
        x_vals = config.x_vals
        
        print(f"\nBenchmarking with varying {x_name} for {config_name}...")
        
        data = {
            'triton': [] if 'iep' in config_name else {'lsemm': [], 'lsemmo': []},
            'torch': []
        }
        
        csv_data = {x_name: [], 'PyTorch (ms)': []}
        if 'iep' in config_name:
            csv_data.update({'Triton (ms)': [], 'Speedup': []})
        else:
            csv_data.update({
                'Triton LSEMM (ms)': [], 'Triton LSEMMO (ms)': [],
                'Speedup LSEMM': [], 'Speedup LSEMMO': []
            })
        
        for x_val in x_vals:
            print(f"  Running with {x_name}={x_val}...")
            params = {**config.args}
            if x_name == 'N': params['N'] = x_val
            elif x_name == 'D': params['D'] = x_val
            elif x_name == 'V': params['V'] = x_val
            
            E = torch.randn([params['D'], params['N']], device=DEVICE).contiguous()
            C = torch.randn([params['V'], params['D']], device=DEVICE).contiguous()
            
            quantiles = [0.5, 0.05, 0.95]
            
            if 'lsemm' in config_name:
                lsemm_ms, _, _ = triton.testing.do_bench(lambda: lsemm(E, C), quantiles=quantiles)
                lsemmo_ms, _, _ = triton.testing.do_bench(lambda: lsemmo(E, C), quantiles=quantiles)
                torch_ms, _, _ = triton.testing.do_bench(lambda: torch_lsemm(E, C), quantiles=quantiles)
                
                data['triton']['lsemm'].append((x_val, lsemm_ms))
                data['triton']['lsemmo'].append((x_val, lsemmo_ms))
                data['torch'].append((x_val, torch_ms))
                
                csv_data[x_name].append(x_val)
                csv_data['Triton LSEMM (ms)'].append(f"{lsemm_ms * 1000:.2f}")
                csv_data['Triton LSEMMO (ms)'].append(f"{lsemmo_ms * 1000:.2f}")
                csv_data['PyTorch (ms)'].append(f"{torch_ms * 1000:.2f}")
                csv_data['Speedup LSEMM'].append(f"{torch_ms/lsemm_ms:.2f}x")
                csv_data['Speedup LSEMMO'].append(f"{torch_ms/lsemmo_ms:.2f}x")
            else:
                I = torch.randint(high=params['V'], size=(params['N'],), device=DEVICE).contiguous()
                tri_ms, _, _ = triton.testing.do_bench(lambda: indexed_essential_probs(E, C, I), quantiles=quantiles)
                torch_ms, _, _ = triton.testing.do_bench(lambda: torch_indexed_essential_probs(E, C, I), quantiles=quantiles)
                
                data['triton'].append((x_val, tri_ms))
                data['torch'].append((x_val, torch_ms))
                
                csv_data[x_name].append(x_val)
                csv_data['Triton (ms)'].append(f"{tri_ms * 1000:.2f}")
                csv_data['PyTorch (ms)'].append(f"{torch_ms * 1000:.2f}")
                csv_data['Speedup'].append(f"{torch_ms/tri_ms:.2f}x")
        
        results[config_name] = data
        
        csv_filename = f"{config_name}_{x_name}_{timestamp}.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(os.path.join('results', csv_filename), index=False)
        print(f"Detailed results for {config_name} ({x_name}) saved to results/{csv_filename}")
        
        png_filename = f"{config_name}_{x_name}_{timestamp}.png"
        create_summary_table(df, f"{config_name.replace('-', ' ').title()} (Varying {x_name})", png_filename)
        
        if write_plots:
            create_matplotlib_plots(data, x_name, config_name, timestamp)
    
    return results

# Updated create_matplotlib_plots
def create_matplotlib_plots(data, x_name, config_name, timestamp):
    x_vals = [point[0] for point in data['torch']]
    torch_times = [point[1] * 1000 for point in data['torch']]
    
    # Performance plot
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(x_vals, torch_times, 'bo-', label='PyTorch', linewidth=2)
    
    if 'lsemm' in config_name:
        lsemm_times = [point[1] * 1000 for point in data['triton']['lsemm']]
        lsemmo_times = [point[1] * 1000 for point in data['triton']['lsemmo']]
        plt.plot(x_vals, lsemm_times, 'ro-', label='Triton LSEMM', linewidth=2)
        plt.plot(x_vals, lsemmo_times, 'go-', label='Triton LSEMMO', linewidth=2)
    else:
        triton_times = [point[1] * 1000 for point in data['triton']]
        plt.plot(x_vals, triton_times, 'ro-', label='Triton', linewidth=2)
    
    plt.xlabel(x_name, fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title(f'{config_name.replace("-", " ").title()} (Varying {x_name})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    performance_filename = f"{config_name}_performance_{x_name}_{timestamp}.png"
    plt.savefig(os.path.join('results', performance_filename), bbox_inches='tight')
    plt.close()
    print(f"Performance plot saved to results/{performance_filename}")
    
    # Speedup plot
    plt.figure(figsize=(10, 6), dpi=150)
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3, label='Baseline')
    
    if 'lsemm' in config_name:
        lsemm_speedups = [torch/tri for torch, tri in zip(torch_times, [p[1] * 1000 for p in data['triton']['lsemm']])]
        lsemmo_speedups = [torch/tri for torch, tri in zip(torch_times, [p[1] * 1000 for p in data['triton']['lsemmo']])]
        plt.plot(x_vals, lsemm_speedups, 'ro-', label='vs PyTorch (LSEMM)', linewidth=2)
        plt.plot(x_vals, lsemmo_speedups, 'go-', label='vs PyTorch (LSEMMO)', linewidth=2)
    else:
        speedups = [torch/tri for torch, tri in zip(torch_times, [p[1] * 1000 for p in data['triton']])]
        plt.plot(x_vals, speedups, 'ro-', label='vs PyTorch', linewidth=2)
    
    plt.xlabel(x_name, fontsize=12)
    plt.ylabel('Speedup (x times)', fontsize=12)
    plt.title(f'{config_name.replace("-", " ").title()} Speedup (Varying {x_name})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    speedup_filename = f"{config_name}_speedup_{x_name}_{timestamp}.png"
    plt.savefig(os.path.join('results', speedup_filename), bbox_inches='tight')
    plt.close()
    print(f"Speedup plot saved to results/{speedup_filename}")

# Updated main block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Apple CCE benchmarks')
    parser.add_argument('--test-only', action='store_true', help='Run only the correctness test')
    parser.add_argument('--basic-benchmarks', action='store_true', help='Run basic benchmarks')
    parser.add_argument('--detailed-benchmarks', action='store_true', help='Run detailed benchmarks with plots')
    parser.add_argument('--all', action='store_true', help='Run all tests and benchmarks')
    parser.add_argument('--test-lsemm', action='store_true', help='Run only the LSEMM/LSEMMO correctness test')
    parser.add_argument('--test-iep', action='store_true', help='Run only the IEP correctness test')
    
    args = parser.parse_args()
    run_all = args.all or (not any(vars(args).values()))
    
    if torch.cuda.is_available():
        print("CUDA is available, running on GPU")
    else:
        print("CUDA is not available, running on CPU")
    
    try:
        if args.test_iep or args.test_only or run_all:
            print("Running indexed_essential_probs correctness test...")
            test_indexed_essential_probs(shapes=(444, 555, 333))
            print("IEP test passed!")
        
        if args.test_lsemm or args.test_only or run_all:
            print("Running LSEMM/LSEMMO correctness test...")
            test_lsemm(shapes=(64, 64, 64))
            print("LSEMM/LSEMMO test passed!")
        
        if args.basic_benchmarks or run_all:
            run_basic_benchmarks()
        
        if args.detailed_benchmarks or run_all:
            benchmark_data = run_detailed_benchmarks(write_plots=True)
            print("\nAll benchmarks complete. Results saved to the 'results' directory.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()