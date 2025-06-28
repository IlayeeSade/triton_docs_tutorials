import torch
import triton
import triton.language as tl
import numpy as np

@triton.jit
def _debug_lsemm_kernel(
    e_ptr, c_ptr, output_ptr, locks_ptr, maxes_ptr,
    N, D, V, L,
    stride_ed, stride_en,
    stride_cv, stride_cd,
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
    block_gmx = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32) # global max of block currently
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
    num_masked_v = tl.sum(1 - (mask_v))
    block_cmx = tl.max(accb, axis=0) # (BN,)
    accb -= block_cmx[None, :]
    acc += tl.sum(tl.exp(accb), axis=0) # (BN,)
    # acc -= num_masked_v * tl.exp(-block_cmx) * tl.cdiv(D, BLOCK_SIZE_D)
    acc = tl.log(acc)

    # Now acc holds log(sum(exp(z_i - block_cmx))) while the real result is
    # log(sum(exp(z_i - block_cmx))) + block_cmx
    
    maxes_ptrs = maxes_ptr + offsets_M * stride_miv
    ointermediate_ptrs = output_ptr + offsets_O * stride_on
    
    mask_m = (offsets_M < N)
    mask_o = (offsets_O < N)

    # count_ptr = locks_ptr + L * num_PID_along_N * stride_ll
    lock_id = PID_N
    l1 = locks_ptr + lock_id * stride_ll
    l2 = locks_ptr + (L + lock_id) * stride_ll

    while tl.atomic_cas(l1, 0, 1) == 1:
        pass

    # Saving useless addition
    # count = tl.load(count_ptr)
    # if count == 0:
    #     tl.atomic_xchg(count_ptr, 1)
    # else:

    # We basically keep the maximum here at all times
    block_gmx = tl.load(maxes_ptrs, mask=mask_m, other=float('-inf'))
    block_gmx = tl.where(block_gmx >= block_cmx, block_gmx, block_cmx)
    tl.store(maxes_ptrs, block_gmx, mask=mask_m) # Store the maxes of the maxes of the block
    
    tl.atomic_add(l2, 1)
    tl.atomic_xchg(l1, 0) # Unlock

    while tl.atomic_cas(l2, num_PID_along_M, 0) != num_PID_along_M:
        pass

    block_gmx = tl.load(maxes_ptrs, mask=mask_m, other=float('-inf'))
    cexpc, cexpg = block_cmx - block_gmx, block_gmx - block_cmx
    keep_mask = block_gmx >= block_cmx
    block_acc = tl.load(ointermediate_ptrs, mask=mask_o) # Holds sum(exp(z_block - block_gmx))
    # block_acc = tl.exp(block_acc) # block_acc holds sum(exp(z_block - block_gmx))
    # sum(exp(z_i - block_cmx) * exp(block_cmx - block_gmx) = sum(exp(z_i - block_cmx + block_cmx - block_gmx)
    corspt = tl.where(keep_mask, block_acc, acc)
    block_acc = tl.where(keep_mask, acc + cexpc, block_acc + cexpg)
    # depending on the mask, (1) if gmx greater/equal, (2) else
    # (1) Now acc holds sum(exp(z_i - block_gmx)) , shape (BLOCK_SIZE_N,)
    # (2) Now block_acc holds sum(exp(z_block - block_cmx)), shape(BLOCK_SIZE_N)
    block_acc = tl.exp(corspt) + tl.exp(block_acc)
    # Now everything is summed and holds the max, not holds, more like holds the effect
    tl.store(ointermediate_ptrs, tl.log(block_acc), mask=mask_o)
    tl.atomic_xchg(l2, num_PID_along_M) # Unlock


def torch_lsemm(E, C):
    assert C.ndim == E.ndim == 2, "only supports matrices, not vectors or tensors"
    assert C.shape[1] == E.shape[0], "incompatible dimensions"
    RES = C @ E  # (V, D) @ (D, N) = (V, N)
    mx = torch.max(RES, dim=0, keepdim=True)[0]  # Shape: (1, N)
    RES = RES - mx  # Broadcasting: (V, N) - (1, N)
    RES = torch.sum(torch.exp(RES), dim=0)  # Sum over V dimension, result shape: (N,)
    return torch.log(RES), mx

def compare_implementations(V=64, D=64, N=64):
    # Random input generation
    E = torch.randn(D, N, dtype=torch.float32)
    C = torch.randn(V, D, dtype=torch.float32)
    
    # Triton kernel parameters
    BLOCK_SIZE_V = min(triton.next_power_of_2(V), 64)
    BLOCK_SIZE_D = min(triton.next_power_of_2(D), 64)
    BLOCK_SIZE_N = min(triton.next_power_of_2(N), 64)

    import math
    # Prepare Triton inputs
    e = E.cuda()
    c = C.cuda()
    O = torch.zeros((N,), device=e.device, dtype=torch.float32)
    M = torch.full((N,), float('-inf'), device=e.device, dtype=torch.float32)
    L = math.ceil(V / BLOCK_SIZE_V)
    locks = torch.zeros(2 * L, dtype=torch.int32, device=e.device)
    
    # Configure grid
    grid = lambda meta: (triton.cdiv(V, meta['BLOCK_SIZE_V']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    # Run Triton kernel
    _debug_lsemm_kernel[grid](
        e, c, O, locks, M,
        N, D, V, L,
        E.stride(1), E.stride(0),
        C.stride(1), C.stride(0), 
        O.stride(0), locks.stride(0), M.stride(0),
        BLOCK_SIZE_N=BLOCK_SIZE_N, 
        BLOCK_SIZE_D=BLOCK_SIZE_D, 
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        GROUP_SIZE=8,
        num_stages=3,
    )
    
    # PyTorch reference
    torch_output, torch_max = torch_lsemm(E, C)
    
    # Compare results
    print("Triton Output:", M.cpu())
    print("PyTorch Output:", torch_max.cpu())
    #print("Max Difference:", torch.max(torch.abs(output.cpu() - torch_output.cpu())))
    
    # Print debug information
    #print("Debug Output:", debug_output.cpu())
    
    #return output, torch_output

# Run the comparison
compare_implementations()