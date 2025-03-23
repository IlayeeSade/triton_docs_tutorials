import torch
import triton
import triton.language as tl

import torch
DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

@triton.jit
def _indexed_essential_probs_kernel(
    e_ptr, i_ptr, # contains the index of the correct label
    c_ptr, output_ptr,
    N, D, V,
    stride_ed, stride_en,
    stride_cv, stride_cd,
    stride_in,
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
    n_dim_offset = pid * BLOCK_SIZE_N
    i_offsets = (n_dim_offset + tl.arange(0, BLOCK_SIZE_N)) * stride_in
    mask_n = i_offsets < N
    cor_idxs = tl.load(i_ptr + i_offsets, mask=mask, dtype=tl.int32) # shape of (BLOCK_SIZE_N,)
    # these indices are according to a tensor C with shape (V, D) 
    # Thus, need to be multiplied by stride_cv
    c_offsets = (cor_idxs * stride_cv)[:, None] + (tl.arange(0, BLOCK_SIZE_D) * stride_cd)[None, :]
    # Offsets of shape (BLOCK_SIZE_N, BLOCK_SIZE_D)
    e_offsets = (tl.arange(0, BLOCK_SIZE_D) * stride_cd)[:, None] + i_offsets[None, :]
    # Offsets of shape (BLOCK_SIZE_D, BLOCK_SIZE_N)
    indices_mask = cor_idxs < N
    mask_d = tl.arange(0, BLOCK_SIZE_D)
    for d in tl.range(0, D, BLOCK_SIZE_D, num_stages=num_stages):
        # Loading C_BLOCK
        mask_c = (indices_mask)[:, None] & (mask_d < D)
        C_BLOCK = tl.load(c_ptr + c_offsets, mask=mask_c)
        # Loading E_BLOCK
        mask_e = (mask_n)[:, None] & (mask_d < D)
        E_BLOCK = tl.load(e_ptr + e_offsets, mask=mask_e)

        c_offsets += BLOCK_SIZE_D * stride_cd
        e_offsets += BLOCK_SIZE_D * stride_ed
        d_blocks_mask += BLOCK_SIZE_D
    

    row_offsets = (m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) * stride_m
    col_offsets = (n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) * stride_n

    offsets = row_offsets[:, None] + col_offsets[None, :] # BLOCK OFFSETS
    mask = (row_offsets[:, None] < m) & (col_offsets[None, :] < n) # MASK SO THAT WE DONT OVERSHOOT


    x = tl.load(x_ptr + offsets, mask=mask) # OTHER DEFAULTS TO 0.0
    # randomly prune it 
    random = tl.rand(seed + m_idx * blocks_in_n + n_idx, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def indexed_essential_probs(E, C, I):
    (D, N) , (V, _) = E.shape, C.shape
    O = torch.empty_like(N)
    assert x.is_contiguous()
    m, n = x.shape
    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_SIZE_N']), )
    _indexed_essential_probs_kernel[grid](
                        E, C, I, O
                        N, D, V,
                        E.stride(0), E.stride(1),
                        C.stride(0), C.stride(1),
                        I.stride(0),
                        BLOCK_SIZE_N=128, BLOCK_SIZE_D=128)
    return O


x = torch.randn(size=(3,3), device=DEVICE)
# Compare this to the baseline - dropout mask is never instantiated!
output = seeded_dropout(x, p=0.5, seed=123)
output2 = seeded_dropout(x, p=0.5, seed=123)
output3 = seeded_dropout(x, p=0.5, seed=512)

print(
        "input", x.tolist(),
        "output (seed = 123)", output.tolist(),
        "output (seed = 123)", output2.tolist(),
        "output (seed = 512)", output3.tolist(),
)