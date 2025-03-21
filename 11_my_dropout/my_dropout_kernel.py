import tabulate
import torch

import triton
import triton.language as tl

import torch
DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    seed,
    m, n,
    p,
    stride_m, stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    blocks_in_n = tl.cdiv(n, BLOCK_SIZE_N)
    m_idx = pid // BLOCK_SIZE_M
    n_idx = pid % BLOCK_SIZE_N

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


def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    m, n = x.shape
    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_SIZE_M']) * triton.cdiv(n, meta['BLOCK_SIZE_N']), )
    _seeded_dropout[grid](
                        x, output, seed, # Arrays
                        m, n,
                        p,
                        x.stride(0), x.stride(1),
                        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128)
    return output


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