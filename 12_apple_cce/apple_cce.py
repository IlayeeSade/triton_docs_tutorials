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
    n_dim_offset = pid * BLOCK_SIZE_N
    i_offsets = (n_dim_offset + tl.arange(0, BLOCK_SIZE_N))
    mask_n = i_offsets < N
    cor_idxs = tl.load(i_ptr + i_offsets * stride_in, mask=mask_n, dtype=tl.int32) # shape of (BLOCK_SIZE_N,)
    # these indices are according to a tensor C with shape (V, D) 
    # Thus, need to be multiplied by stride_cv
    c_offsets = (cor_idxs * stride_cv)[:, None] + (tl.arange(0, BLOCK_SIZE_D) * stride_cd)[None, :]
    # Offsets of shape (BLOCK_SIZE_N, BLOCK_SIZE_D)
    e_offsets = (tl.arange(0, BLOCK_SIZE_D) * stride_ed)[:, None] + i_offsets[None, :] * stride_en
    # Offsets of shape (BLOCK_SIZE_D, BLOCK_SIZE_N)
    indices_mask = cor_idxs < N
    mask_d = tl.arange(0, BLOCK_SIZE_D)

    acc = tl.zeros((N,), dtype=tl.float32)
    for d in tl.range(0, D, BLOCK_SIZE_D, num_stages=num_stages):
        # Loading C_BLOCK
        mask_c = (indices_mask)[:, None] & (mask_d < D)[None, :]
        C_BLOCK = tl.load(c_ptr + c_offsets, mask=mask_c)
        # Loading E_BLOCK
        mask_e = (mask_d < D)[None, :] & (mask_n)[:, None]
        E_BLOCK = tl.load(e_ptr + e_offsets, mask=mask_e)

        acc += tl.sum((C_BLOCK * tl.trans(E_BLOCK)), axis=1)
        # We calclate dot product for every E_i , C_i

        c_offsets += BLOCK_SIZE_D * stride_cd
        e_offsets += BLOCK_SIZE_D * stride_ed
        mask_d += BLOCK_SIZE_D

    tl.store(output_ptr + i_offsets * stride_on, acc, mask=mask_n)


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
                        I.stride(0), O.stride(0)
                        BLOCK_SIZE_N=128, BLOCK_SIZE_D=128)
    return O


def torch_indexed_essential_probs(E, C, I, use_vectorized=True):
    N = I.shape[0]
    if use_vectorized:
        indexed_C = C[I]
        E_t = E.T
        O = (indexed_C * E_t).sum(dim=1)
    else:
        for i in range(N):
            idx = I[i]
            c_row = C[idx]
            # Get the i-th column from E
            e_col = E[:, i]
            O[i] = torch.dot(c_row, e_col)
    return O

def test_indexed_essential_probs(shapes: tuple, atol=1e-2, rtol=1e-1, device=DEVICE):
    # create input data
    torch.manual_seed(0)
    assert type(shapes) == tuple and len(shapes) == 3
    N, D, V = shapes
    E, C, O, I = torch.randn([D, N]), torch.randn([V, D]), torch.empty([N]), torch.randint(high=V, size=(N,))

    a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    # run kernel & pytorch reference implementation
    c_tri = indexed_essential_probs(E, C, I)
    c_ref = torch_indexed_essential_probs(E, C, I)
    # compare
    torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
    print("PASSED")

if __name__ == "__main__":
    test_indexed_essential_probs(shapes=(1024, 1024, 1024))