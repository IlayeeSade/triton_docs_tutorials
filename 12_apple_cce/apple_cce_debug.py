import torch
import triton
import triton.language as tl
import math

# Use a single, clear list of configs
autotune_configs_lsemm = [
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_D': 64, 'BLOCK_SIZE_V': 64, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=4),
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_D': 32, 'BLOCK_SIZE_V': 32, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=4),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_D': 32, 'BLOCK_SIZE_V': 32, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=4),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_D': 64, 'BLOCK_SIZE_V': 64, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=8),
    triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_D': 64, 'BLOCK_SIZE_V': 64, 'GROUP_SIZE': 8, 'num_stages': 3}, num_warps=8),
    # Consider adding more diverse configs if needed, e.g., different num_stages/num_warps
]

@triton.autotune(configs=autotune_configs_lsemm, key=['N', 'D', 'V'])
@triton.jit
def _lsemm_kernel(
    e_ptr, c_ptr, output_ptr, locks_ptr, maxes_ptr,
    N, D, V,
    # L is not needed inside the kernel if locks_ptr has size 2*N
    stride_cv, stride_cd,
    stride_ed, stride_en,
    stride_on, stride_ll,
    stride_miv,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_V: tl.constexpr,
    GROUP_SIZE: tl.constexpr,  num_stages: tl.constexpr,
):
    # We want to matmul (V, D) @ (D, N) = (V, N)
    # Then apply log(sum(exp(X - max(X)))) + max(X) reduction along V axis
    PID = tl.program_id(axis=0)

    # Calculate PID_N and PID_M (V dimension) based on group-major ordering
    num_PID_along_M = tl.cdiv(V, BLOCK_SIZE_V) # Number of blocks along V
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N) # Number of blocks along N
    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    group_id = PID // num_PID_in_group
    first_PID_in_group_along_M = group_id * GROUP_SIZE
    # Adjust group size for the last group along M if V is not perfectly divisible
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE)

    # Check if group_size_adj is zero (can happen with empty last groups)
    # Although unlikely with typical grid calculation, better safe than sorry.
    if group_size_adj == 0:
        return # This block is outside the valid range

    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
    PID_N = (PID % num_PID_in_group) // group_size_adj # Index along N dimension blocks

    # --- Block Pointer Setup ---
    offsets_V = PID_M * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V) # (BLOCK_SIZE_V,)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) # (BLOCK_SIZE_N,)
    offsets_D = tl.arange(0, BLOCK_SIZE_D) # (BLOCK_SIZE_D,)

    # Pointers for C (shape V, D) -> A for matmul (transposed conceptually)
    a_ptr = c_ptr + offsets_V[:, None] * stride_cv + offsets_D[None, :] * stride_cd # (BV, BD)
    # Pointers for E (shape D, N) -> B for matmul
    b_ptr = e_ptr + offsets_D[:, None] * stride_ed + offsets_N[None, :] * stride_en # (BD, BN)

    # --- Initialize Accumulators ---
    # Accumulator for C @ E result block
    acc_matmul = tl.zeros((BLOCK_SIZE_V, BLOCK_SIZE_N,), dtype=tl.float32)

    # --- Inner Loop over D ---
    # Use range and cdiv for robust loop bounds
    for k in range(0, tl.cdiv(D, BLOCK_SIZE_D)):
        # Boundary masks
        mask_v = offsets_V[:, None] < V # (BV, 1)
        mask_n = offsets_N[None, :] < N # (1, BN)
        # Current K-block dimension check
        mask_d_a = (k * BLOCK_SIZE_D + offsets_D[None,:]) < D # (1, BD)
        mask_d_b = (k * BLOCK_SIZE_D + offsets_D[:,None]) < D # (BD, 1)

        mask_a = mask_v & mask_d_a # (BV, BD)
        mask_b = mask_d_b & mask_n # (BD, BN)

        a = tl.load(a_ptr, mask=mask_a, other=0.0)
        b = tl.load(b_ptr, mask=mask_b, other=0.0)

        # Matrix multiplication
        acc_matmul = tl.dot(a, b, acc=acc_matmul)

        # Advance pointers for next K block
        a_ptr += BLOCK_SIZE_D * stride_cd
        b_ptr += BLOCK_SIZE_D * stride_ed

    # --- LogSumExp Reduction within Block ---
    # Mask for valid V and N elements in the result block
    mask_vn = (offsets_V[:, None] < V) & (offsets_N[None, :] < N) # (BV, BN)

    # Find block maximum along V axis, considering only valid N columns
    # Initialize max with -inf where N is invalid
    block_max = tl.max(tl.where(mask_vn, acc_matmul, float('-inf')), axis=0) # (BN,)
    # Subtract block maximum
    acc_matmul = acc_matmul - block_max[None, :] # (BV, BN)
    # Compute exp, masking out invalid elements (set exp(-inf) -> 0)
    exp_values = tl.exp(tl.where(mask_vn, acc_matmul, float('-inf'))) # (BV, BN)
    # Sum exp values along V axis
    block_sum_exp = tl.sum(exp_values, axis=0) # (BN,)

    # --- Atomic Update for Global Sum and Max ---
    # Pointers to output sum (O) and max (M) for this N-block
    # Note: offsets_N are the correct offsets for O and M which have shape (N,)
    output_block_ptr = output_ptr + offsets_N * stride_on
    maxes_block_ptr = maxes_ptr + offsets_N * stride_miv

    # Mask for valid N columns in this block
    mask_n_update = offsets_N < N # (BN,)

    # Use PID_N as the lock ID (one lock per output N-block)
    lock_id = PID_N
    lock_ptr = locks_ptr + lock_id * stride_ll
    # Use a dedicated counter for first-write detection (safer than checking output)
    # Stored after the primary locks, starting at index N
    count_ptr = locks_ptr + (N + lock_id) * stride_ll

    # Acquire lock
    while tl.atomic_cas(lock_ptr, 0, 1) == 1:
        pass
    # --- Critical Section ---
    count = tl.load(count_ptr) # Check if this N-block has been written to before

    if count == 0:
        # First writer for this N-block
        tl.atomic_xchg(count_ptr, 1) # Mark as written
        # Store initial sum and max
        tl.store(output_block_ptr, block_sum_exp, mask=mask_n_update)
        tl.store(maxes_block_ptr, block_max, mask=mask_n_update)
    else:
        # Combine with existing values
        current_sum = tl.load(output_block_ptr, mask=mask_n_update) # sum(exp(z_old - max_old))
        # Atomically update the global max for this block, getting the OLD max value
        old_max = tl.atomic_max(maxes_block_ptr, block_max, mask=mask_n_update)
        # Load the NEW global max (which might be old_max or block_max)
        new_max = tl.load(maxes_block_ptr, mask=mask_n_update)

        # Scaling factors to align sums to the new_max
        # scale_new = exp(block_max - new_max)
        # scale_old = exp(old_max - new_max)
        # Avoid potential exp(0) issues if maxes are equal; use precise calculation
        scale_new = tl.exp(block_max - new_max)
        scale_old = tl.exp(old_max - new_max)

        # Combine sums: sum_new * scale_new + sum_old * scale_old
        combined_sum = block_sum_exp * scale_new + current_sum * scale_old
        # Store the combined sum (relative to new_max)
        tl.store(output_block_ptr, combined_sum, mask=mask_n_update)

    # --- End Critical Section ---
    # Release lock
    tl.atomic_xchg(lock_ptr, 0)


# Wrapper function (corrected version, similar to your original lsemm)
def lsemm_triton(E, C):
    assert C.ndim == E.ndim == 2, "only supports matrices, not vectors or tensors"
    assert C.shape[1] == E.shape[0], "incompatible dimensions"
    assert C.is_cuda and E.is_cuda, "Inputs must be CUDA tensors"
    assert C.dtype == E.dtype, "Inputs must have the same dtype"
    # Consider adding support for float16/bfloat16 if needed

    (D, N), (V, _) = E.shape, C.shape

    # Output for sums: sum(exp(value - final_max)) -> initialize to 0.0
    O = torch.zeros((N,), device=E.device, dtype=torch.float32)
    # Output for final maxes -> initialize to -inf
    M = torch.full((N,), float('-inf'), device=E.device, dtype=torch.float32)

    # Locks: Need N locks + N counters = 2 * N
    # One lock per N-block column (index 0 to N-1)
    # One counter per N-block column (index N to 2N-1)
    num_locks = N
    locks = torch.zeros(num_locks * 2, dtype=torch.int32, device=E.device)

    grid = lambda meta: (triton.cdiv(V, meta['BLOCK_SIZE_V']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )

    _lsemm_kernel[grid](
        E, C, O, locks, M,
        N, D, V,
        C.stride(0), C.stride(1), # stride_cv, stride_cd
        E.stride(0), E.stride(1), # stride_ed, stride_en
        O.stride(0), locks.stride(0), M.stride(0), # stride_on, stride_ll, stride_miv
        # BLOCK_SIZE_N, D, V passed via autotuner
        # GROUP_SIZE, num_stages passed via autotuner
    )

    # Final calculation: log(sum(exp(value - final_max))) + final_max
    # Need to handle O=0 case (log(0) = -inf), happens if all inputs were -inf
    # Add small epsilon? Or rely on log(+0) = -inf which might be okay.
    # Let's compute log carefully
    # Use torch.log1p on O-1 if O is near 1? No, O is sum_exp.
    # If O is zero, the log is -inf. If O is positive, log is finite.
    final_out = torch.log(O + 1e-20) + M # Add epsilon for numerical stability if O can be exactly 0

    # If the true result could be -inf (e.g. all inputs -inf),
    # handle cases where O becomes 0 and M is -inf.
    # torch.log(0) = -inf. -inf + (-inf) = -inf. Seems okay.

    return final_out

@torch.compile
def torch_lsemm(E, C):
    # Use float32 for intermediate calculations for better precision comparison
    C_f32 = C.to(torch.float32)
    E_f32 = E.to(torch.float32)

    RES = C_f32 @ E_f32  # (V, D) @ (D, N) = (V, N)
    mx = torch.max(RES, dim=0, keepdim=True)[0]  # Shape: (1, N)
    # Handle case where mx is -inf (all inputs -inf)
    mx = torch.where(torch.isneginf(mx), 0.0, mx) # Replace -inf max with 0, exp(x - 0) = exp(x)
    RES = RES - mx  # Broadcasting: (V, N) - (1, N)
    RES_sum = torch.sum(torch.exp(RES), dim=0)  # Sum over V dimension, result shape: (N,)
    # Add epsilon before log for stability
    out = torch.log(RES_sum + 1e-20) + mx.squeeze(0) # Squeeze max back to (N,)
    return out


# --- Testing ---
DEVICE = 'cuda'

def test_lsemm(shapes: tuple, dtype=torch.float32, atol=1e-2, rtol=1e-1, device=DEVICE):
    torch.manual_seed(0)
    assert type(shapes) == tuple and len(shapes) == 3
    N, D, V = shapes
    # Use reasonably scaled inputs
    E = torch.randn([D, N], device=device, dtype=dtype) * 5
    C = torch.randn([V, D], device=device, dtype=dtype) * 5

    # Test Triton implementation
    try:
        c_tri_lsemm = lsemm_triton(E, C)
    except Exception as e:
        print(f"Triton kernel failed: {e}")
        # Optionally dump inputs/outputs or raise
        raise

    # Reference PyTorch implementation
    c_ref = torch_lsemm(E, C)

    # Compare results
    print(f"\n--- Testing Shapes N={N}, D={D}, V={V} ---")
    print("Triton LSEMM (max):", c_tri_lsemm.max().item())
    print("PyTorch LSEMM (max):", c_ref.max().item())
    print("Triton LSEMM (sample):", c_tri_lsemm[:5])
    print("PyTorch LSEMM (sample):", c_ref[:5])

    # --- Adjust Tolerances ---
    # Floating point differences ARE expected due to non-associativity.
    # Tolerances might need to be looser than typical exact matmul tests.
    # Start with the user's tolerances and potentially adjust based on observation.
    print(f"Using atol={atol}, rtol={rtol}")
    try:
        torch.testing.assert_close(c_tri_lsemm, c_ref, atol=atol, rtol=rtol)
        print("✅ PASSED - Triton implementation matches PyTorch reference within tolerance.")
    except AssertionError as e:
        print(f"❌ FAILED - Mismatch detected!")
        print(e)
        # Optionally, calculate and print max difference
        abs_diff = torch.abs(c_tri_lsemm - c_ref)
        rel_diff = abs_diff / torch.abs(c_ref)
        print(f"Max Absolute Difference: {torch.max(abs_diff).item()}")
        print(f"Max Relative Difference: {torch.max(rel_diff).item()}")

# Example Usage:
if __name__ == "__main__":
    test_lsemm((1024, 512, 2048)) # N, D, V
    test_lsemm((256, 128, 512))
    test_lsemm((4096, 64, 4096))
    # Add a case with non-power-of-2 dimensions
    test_lsemm((1000, 500, 2000))