import torch
import triton
import triton.language as tl
import numpy as np

@triton.jit
def _debug_lsemm_kernel(
    e_ptr, c_ptr, output_ptr, locks_ptr, maxes_ptr, p_ptr,
    N, D, V, L,
    stride_cv, stride_cd,
    stride_ed, stride_en,
    stride_on, stride_ll,
    stride_miv,
    stride_pv, stride_pn,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_V: tl.constexpr,
    GROUP_SIZE: tl.constexpr,  num_stages: tl.constexpr,
):
    # We want to matmul (V, D) @ (D, N) and the sum over the V axis
    PID = tl.program_id(axis=0) 
    
    PID_V, PID_N = PID % V, PID // V

    offsets_V = PID_V * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_D = tl.arange(0, BLOCK_SIZE_D)
    
    # Reference
    offsets_O = offsets_N
    offsets_M = offsets_N
    
    
    a_offsets = offsets_V[:, None] * stride_cv + offsets_D[None, :] * stride_cd # (BV, BD)
    b_offsets = offsets_D[:, None] * stride_ed + offsets_N[None, :] * stride_en # (BD, BN)

    acc = tl.zeros((BLOCK_SIZE_V, BLOCK_SIZE_N,), dtype=tl.float32)
    block_cmx = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32) # current max

    mask_v = offsets_V < V
    mask_n = offsets_N < N
        
    for d in range(0, tl.cdiv(D, BLOCK_SIZE_D), num_stages=num_stages):
        mask_d = offsets_D < D - d * BLOCK_SIZE_D
        mask_a = mask_v[:, None] & mask_d[None, :]
        mask_b = mask_d[:, None] & mask_n[None, :]

        a = tl.load(c_ptr + a_offsets, mask=mask_d[None, :], other=0.0)
        b = tl.load(e_ptr + b_offsets, mask=mask_d[:, None], other=0.0)
        
        # a @ b => (BV, BN) and we need to sum over BV
        acc = tl.dot(a, b, acc=acc) # (BLOCK_SIZE_V, BLOCK_SIZE_N)
    
        a_offsets += BLOCK_SIZE_D * stride_cd
        b_offsets += BLOCK_SIZE_D * stride_ed


    offsets_P = (offsets_V * stride_pv)[:, None] + (offsets_N * stride_pn)[None, :]
    mask_p = mask_v[:, None] & mask_n[None, :]
    tl.store(p_ptr + offsets_P, acc, mask=mask_p)

    block_cmx = tl.max(acc, axis=0) # (BN,)
    maxes_ptrs = maxes_ptr + offsets_M * stride_miv
    mask_m = (offsets_M < N)

    tl.atomic_max(maxes_ptrs, block_cmx, mask_m)


def torch_lsemm(E, C):
    assert C.ndim == E.ndim == 2, "only supports matrices, not vectors or tensors"
    assert C.shape[1] == E.shape[0], "incompatible dimensions"
    RES = C @ E  # (V, D) @ (D, N) = (V, N)
    mx = torch.max(RES, dim=0, keepdim=True)[0]  # Shape: (1, N)
    return RES, mx

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
    L = math.ceil(N / BLOCK_SIZE_N)
    P = torch.empty((V, N,), device=e.device)
    locks = torch.zeros(2 * L, dtype=torch.int32, device=e.device)
    
    # Configure grid
    grid = lambda meta: (triton.cdiv(V, meta['BLOCK_SIZE_V']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    # Run Triton kernel
    _debug_lsemm_kernel[grid](
        e, c, O, locks, M, P,
        N, D, V, L,
        C.stride(0), C.stride(1), 
        E.stride(0), E.stride(1),
        O.stride(0), locks.stride(0), M.stride(0),
        P.stride(0), P.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N, 
        BLOCK_SIZE_D=BLOCK_SIZE_D, 
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        GROUP_SIZE=8,
        num_stages=3,
    )
    
    # PyTorch reference
    torch_output, torch_max = torch_lsemm(E, C)
    
    # Compare results
    print("Triton Output:", P.cpu())
    print("PyTorch Output:", torch_output.cpu())
    #print("Max Difference:", torch.max(torch.abs(output.cpu() - torch_output.cpu())))
    
    # Print debug information
    #print("Debug Output:", debug_output.cpu())
    
    #return output, torch_output

# Run the comparison
compare_implementations()