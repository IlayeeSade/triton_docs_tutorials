"""
This matmul kernel can be a bit confusing but is very crucial to understand

 What you'll learn:
- Automatic performance tuning
- Program re-ordering for improved SRAM hit rate
- Multi-dimensional pointer arithmetic
- High precision data type accumulation
- using the Triton interpreter (kind of)

Recommended order to read the code in:
Step 1 - unit test
Step 2 - wrapper
Step 3 - kernel
Step 4 - benchmark

For matmul of A @ B = C of shapes (M, K) @ (K, N) = (M, N), the following
algorithm is numerically equivalent to what our code will output, but we'll
get to the answer in a different way
for m in range(0, M, BLOCK_SIE_M): # do in parallel
    for n in range(0, N, BLOCK_SIZE_N): # do in parallel
        acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
        for k in range(0, K, BLOCK_SIZE_K):
            a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
            b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
            acc += dot(a,b)
        C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc

see original
https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
"""
import torch
import triton
import triton.language as tl
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

# Configuration system for choosing which kernels to test and benchmark
KERNEL_CONFIG = {
    'test_kernels': ['rbf1', 'rbf2'],  # List of kernels to test: 'rbf1', 'rbf2', 'rbf_custom'
    'benchmark_kernels': ['rbf1', 'rbf2', 'torch'],  # List of kernels to benchmark: 'rbf1', 'rbf2', 'rbf_custom', 'torch'
    'test_sizes': [(512, 512), (1024, 1024), (2048, 2048)],  # Sizes to test
    'benchmark_sizes': [128 * i for i in range(2, 33)],  # Sizes to benchmark
}

def update_kernel_config(test_kernels=None, benchmark_kernels=None, test_sizes=None, benchmark_sizes=None):
    """Update the kernel configuration for testing and benchmarking."""
    global KERNEL_CONFIG
    if test_kernels is not None:
        KERNEL_CONFIG['test_kernels'] = test_kernels
    if benchmark_kernels is not None:
        KERNEL_CONFIG['benchmark_kernels'] = benchmark_kernels
    if test_sizes is not None:
        KERNEL_CONFIG['test_sizes'] = test_sizes
    if benchmark_sizes is not None:
        KERNEL_CONFIG['benchmark_sizes'] = benchmark_sizes

######### Step 3 #########

# un-comment this to run a numpy emulation of Triton on CPU & be able to debug with print() statements
#import os
#os.environ["TRITON_INTERPRET"] = "1"

# autotuning is just setting up a bunch of different potential meta-parameters configurations that Triton will automatically
# choose from later based on which one performs best on our specific GPU. Triton will figure out for us which one to use. They're 
# all values chosen heuristically, but notice everything is a multiple of 32 in sticking w/ the number of threads in a warp.
autotune_configs = [
    # Balanced, medium-sized tiles for versatility
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=3, num_warps=4),
    # Larger tiles for big matrices, high occupancy
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=2, num_warps=8),
    # Tall tiles for large M, smaller N
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 4}, num_stages=3, num_warps=4),
    # Wide tiles for large N, smaller M
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 4}, num_stages=3, num_warps=4),
    # Smaller tiles for resource-constrained cases or small problems
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE': 4}, num_stages=4, num_warps=2),
    # High compute, deeper pipeline for large K
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 4}, num_stages=4, num_warps=8),
    # NEW: Large block size configurations for high-performance scenarios
    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE': 8}, num_stages=2, num_warps=16),
    triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=2, num_warps=16),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=2, num_warps=16),
    # NEW: Very large block sizes for maximum throughput on large matrices
    triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE': 16}, num_stages=1, num_warps=32),
    triton.Config({'BLOCK_SIZE_M': 1024, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE': 16}, num_stages=1, num_warps=32),
]
# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator which consumes
#   1) a list of `triton.Config` objects that define different configs of meta-parameters and compilation options
#   2) an auto-tuning *key* whose change in values will trigger a new evaluation of all the provided configs, meaning
#       that any time either M, N, or K changes with a new input, Triton will check which config is best all over again
@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def _rbf1_kernel(
    a_ptr, b_ptr, c_ptr, 
    M, N, K, 
    stride_a_M, stride_a_K, 
    stride_b_K, stride_b_N, 
    stride_c_M, stride_c_N, 
    # meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, 
):
    # we start with a 1D launch grid that we will turn into a 2D grid with a complicated "group-wise" ordering
    PID = tl.program_id(axis=0) 
    # defining the size of groups
    num_PID_along_M = tl.cdiv(M, BLOCK_SIZE_M) # the number of blocks along M dimension
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N) # the number of blocks along N dimension
    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    # figurinig out which group this PID is in
    group_id = PID // num_PID_in_group 
    # tells us which row to start at for this group
    first_PID_in_group_along_M = group_id * GROUP_SIZE 
    # this is usually equal to GROUP_SIZE; the alternative case happens when we're at edge of the tensor and 
    #  its dimensions don't cleanly divde into GROUP_SIZE # TODO is this true?
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE) 
    # this is the bulk of the actual mapping of PIDs to group-major ordering
    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
        # (PID % num_PID_in_group) puts the current program id into the context of a group
        # (first_PID_in_group_along_m + ...) shifts the PID into the correct group
        # (... % group_size_adj) removes the column component to get us onto the correct row
    PID_N = (PID % num_PID_in_group) // group_size_adj
        # (... // group_size_adj) removes the row component to get us onto the correct column
    
    # Now that the PID nightmare is done we can move onto the kernel code you're more used to seeing.

    # Let's create pointer vectors for the first group of blocks of the input matrices
    offsets_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_K = tl.arange(0, BLOCK_SIZE_K)
    a_offsets = offsets_M[:, None] * stride_a_M + offsets_K[None, :] * stride_a_K
    b_offsets = offsets_N[:, None] * stride_b_N + offsets_K[None, :] * stride_b_K
    """
    [:, None] turns [m1,m2,m3] into [[m1],[m2],[m3]] 
    [None, :] turns [n1,n2,n3] into [[n1,n2,n3]]
    combining them gives the matrix
    [[m1n1, m1n2, m1n3],
     [m2n1, m2n2, m2n3],
     [m3n1, m3n2, m3n3]] 
    """

    # inputs tensors are fp16 but we accumulate into a block of fp32 values for higher accuracy (we'll revert later)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32) # the full C is shape (M, N)
        # for a demonstration of why accumulation works, check out `./block_wise_matmul.png`
        
    # we'll iterate along the K dimension of both A and B to compute a single block of the C matrix
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # out-of-bounds entries (along K) need to be masked out
        mask = offsets_K < K - k * BLOCK_SIZE_K
            # k * BLOCK_SIZE_K is the current starting index of offsets_k.
            # so this only really activates when k is within BLOCK_SIZE_K entries from K
            # meaning this gets triggered on the last iteration of the loop, and only if K is not a multiple of BLOCK_SIZE_K
        
        # Now we load blocks of A and B matrices. If multiple blocks in a group are on the same SM, 
        # they can share these loaded values, which reduces the number of expensive loads from DRAM
        a = tl.load(a_ptr + a_offsets, mask=mask[None, :], other=0.0) # shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptr + b_offsets, mask=mask[None, :], other=0.0) # shape (BLOCK_SIZE_N, BLOCK_SIZE_K)
            # fill in any masked-out parts with 0.0's since they don't have any effect on the summation in the next step

        # we accumulate along the K dimension
        dist_matrix = (a[:, None, :] - b[None, :, :])
        accumulator += tl.sum(dist_matrix * dist_matrix, axis=2)
            # triton is weird with operation notation; this is actually a tiny matmul not a dot product
            #   shape (BLOCK_SIZE_M, BLOCK_SIZE_K) @ (BLOCK_SIZE_K, BLOCK_SIZE_N) = (BLOCK_SIZE_M, BLOCK_SIZE_N)
            # `acc` tells Triton to write the output of the matmul directly to accumulator, which is more efficient than
            #   accumulator += tl.dot(a, b)

        # advance the pointers to the next block along K
        a_offsets += BLOCK_SIZE_K * stride_a_K
        b_offsets += BLOCK_SIZE_K * stride_b_K

    # write back the block of the output matrix C with masks
    c_offsets = stride_c_M * offsets_M[:, None] + stride_c_N * offsets_N[None, :]
    c_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N) # notice the 2D mask
    tl.store(c_ptr + c_offsets, tl.exp(-accumulator).to(tl.float16), mask=c_mask) # shape (BLOCK_SIZE_M, BLOCK_SIZE_N)

@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def _rbf2_kernel(
    a_ptr, b_ptr, c_ptr, 
    M, N, K, 
    stride_a_M, stride_a_K, 
    stride_b_K, stride_b_N, 
    stride_c_M, stride_c_N, 
    # meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, 
):
    # we start with a 1D launch grid that we will turn into a 2D grid with a complicated "group-wise" ordering
    PID = tl.program_id(axis=0) 
    # defining the size of groups
    num_PID_along_M = tl.cdiv(M, BLOCK_SIZE_M) # the number of blocks along M dimension
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N) # the number of blocks along N dimension
    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    # figurinig out which group this PID is in
    group_id = PID // num_PID_in_group 
    # tells us which row to start at for this group
    first_PID_in_group_along_M = group_id * GROUP_SIZE 
    # this is usually equal to GROUP_SIZE; the alternative case happens when we're at edge of the tensor and 
    #  its dimensions don't cleanly divde into GROUP_SIZE # TODO is this true?
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE) 
    # this is the bulk of the actual mapping of PIDs to group-major ordering
    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
        # (PID % num_PID_in_group) puts the current program id into the context of a group
        # (first_PID_in_group_along_m + ...) shifts the PID into the correct group
        # (... % group_size_adj) removes the column component to get us onto the correct row
    PID_N = (PID % num_PID_in_group) // group_size_adj
        # (... // group_size_adj) removes the row component to get us onto the correct column
    
    # Now that the PID nightmare is done we can move onto the kernel code you're more used to seeing.

    # Let's create pointer vectors for the first group of blocks of the input matrices
    offsets_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_K = tl.arange(0, BLOCK_SIZE_K)
    a_offsets = offsets_M[:, None] * stride_a_M + offsets_K[None, :] * stride_a_K 
    # BM x BK
    b_offsets = offsets_N[:, None] * stride_b_N + offsets_K[None, :] * stride_b_K
    # BN x BK

    # inputs tensors are fp16 but we accumulate into a block of fp32 values for higher accuracy (we'll revert later)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32) # the full C is shape (M, N)
        # for a demonstration of why accumulation works, check out `./block_wise_matmul.png`


    # prefetching  
    end = tl.cdiv(K, BLOCK_SIZE_K)
    mask = offsets_K < K
    a_next = tl.load(a_ptr + a_offsets, mask=mask[None, :], other=0.0) # shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
    b_next = tl.load(b_ptr + b_offsets, mask=mask[None, :], other=0.0) # shape (BLOCK_SIZE_N, BLOCK_SIZE_K)
    # we'll iterate along the K dimension of both A and B to compute a single block of the C matrix
    for k in range(0, end):
        # out-of-bounds entries (along K) need to be masked out
        mask = offsets_K < K - k * BLOCK_SIZE_K
            # k * BLOCK_SIZE_K is the current starting index of offsets_k.
            # so this only really activates when k is within BLOCK_SIZE_K entries from K
            # meaning this gets triggered on the last iteration of the loop, and only if K is not a multiple of BLOCK_SIZE_K
        a = a_next
        b = b_next
        if k + 1 != end:
            a_next = tl.load(a_ptr + a_offsets, mask=mask[None, :], other=0.0) # shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
            b_next = tl.load(b_ptr + b_offsets, mask=mask[None, :], other=0.0) # shape (BLOCK_SIZE_N, BLOCK_SIZE_K)
            # fill in any masked-out parts with 0.0's since they don't have any effect on the summation in the next step
        
        # we accumulate along the K dimension
        # || x - y || ^ 2 = || x || ^ 2 + || y || ^ 2 - 2<x,y>
        # a ** 2 (m, k) , summing over k-second dim.
        accumulator -= (tl.sum(a * a, axis=1)[:, None] + tl.sum(b * b, axis=1)[None, :]) # Modifying shapes to broadcast to BM x BN
        accumulator = tl.dot(a, tl.trans(b), acc=accumulator) #( BM X BK @ BK X BN ) X 2

        # advance the pointers to the next block along K
        a_offsets += BLOCK_SIZE_K * stride_a_K   # BM X BK X 4 
        b_offsets += BLOCK_SIZE_K * stride_b_K   # BK X BN X 4

    # write back the block of the output matrix C with masks
    c_offsets = stride_c_M * offsets_M[:, None] + stride_c_N * offsets_N[None, :]
    c_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N) # notice the 2D mask
    tl.store(c_ptr + c_offsets, tl.exp(accumulator).to(tl.float16), mask=c_mask) # shape (BLOCK_SIZE_M, BLOCK_SIZE_N)


@triton.jit
def _rbf_custom_kernel(
    a_ptr, b_ptr, c_ptr, 
    M, N, K, 
    stride_a_M, stride_a_K, 
    stride_b_K, stride_b_N, 
    stride_c_M, stride_c_N, 
    # meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, 
):
    # we start with a 1D launch grid that we will turn into a 2D grid with a complicated "group-wise" ordering
    PID = tl.program_id(axis=0) 
    # defining the size of groups
    num_PID_along_M = tl.cdiv(M, BLOCK_SIZE_M) # the number of blocks along M dimension
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N) # the number of blocks along N dimension
    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    # figurinig out which group this PID is in
    group_id = PID // num_PID_in_group 
    # tells us which row to start at for this group
    first_PID_in_group_along_M = group_id * GROUP_SIZE 
    # this is usually equal to GROUP_SIZE; the alternative case happens when we're at edge of the tensor and 
    #  its dimensions don't cleanly divde into GROUP_SIZE # TODO is this true?
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE) 
    # this is the bulk of the actual mapping of PIDs to group-major ordering
    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
        # (PID % num_PID_in_group) puts the current program id into the context of a group
        # (first_PID_in_group_along_m + ...) shifts the PID into the correct group
        # (... % group_size_adj) removes the column component to get us onto the correct row
    PID_N = (PID % num_PID_in_group) // group_size_adj
        # (... // group_size_adj) removes the row component to get us onto the correct column
    
    # Now that the PID nightmare is done we can move onto the kernel code you're more used to seeing.

    # Let's create pointer vectors for the first group of blocks of the input matrices
    offsets_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_K = tl.arange(0, BLOCK_SIZE_K)
    a_offsets = offsets_M[:, None] * stride_a_M + offsets_K[None, :] * stride_a_K 
    # BM x BK
    b_offsets = offsets_N[:, None] * stride_b_N + offsets_K[None, :] * stride_b_K
    # BN x BK

    # inputs tensors are fp16 but we accumulate into a block of fp32 values for higher accuracy (we'll revert later)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32) # the full C is shape (M, N)
        # for a demonstration of why accumulation works, check out `./block_wise_matmul.png`


    # prefetching  
    end = tl.cdiv(K, BLOCK_SIZE_K)
    mask = offsets_K < K
    a_next = tl.load(a_ptr + a_offsets, mask=mask[None, :], other=0.0) # shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
    b_next = tl.load(b_ptr + b_offsets, mask=mask[None, :], other=0.0) # shape (BLOCK_SIZE_N, BLOCK_SIZE_K)
    # we'll iterate along the K dimension of both A and B to compute a single block of the C matrix
    for k in range(0, end):
        # out-of-bounds entries (along K) need to be masked out
        mask = offsets_K < K - k * BLOCK_SIZE_K
            # k * BLOCK_SIZE_K is the current starting index of offsets_k.
            # so this only really activates when k is within BLOCK_SIZE_K entries from K
            # meaning this gets triggered on the last iteration of the loop, and only if K is not a multiple of BLOCK_SIZE_K
        a = a_next
        b = b_next
        if k + 1 != end:
            a_next = tl.load(a_ptr + a_offsets, mask=mask[None, :], other=0.0) # shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
            b_next = tl.load(b_ptr + b_offsets, mask=mask[None, :], other=0.0) # shape (BLOCK_SIZE_N, BLOCK_SIZE_K)
            # fill in any masked-out parts with 0.0's since they don't have any effect on the summation in the next step
        
        # we accumulate along the K dimension
        # || x - y || ^ 2 = || x || ^ 2 + || y || ^ 2 - 2<x,y>
        # a ** 2 (m, k) , summing over k-second dim.
        a_norm = tl.sum(a * a, axis=1)
        b_norm = tl.sum(b * b, axis=1)
        accumulator -= tl.sum(a * a, axis=1)[:, None]
        accumulator -= tl.sum(b * b, axis=1)[None, :]
        accumulator = tl.dot(a, tl.trans(b), acc=accumulator)
        # triton is weird with operation notation; this is actually a tiny matmul not a dot product
        #   shape (BLOCK_SIZE_M, BLOCK_SIZE_K) @ (BLOCK_SIZE_K, BLOCK_SIZE_N) = (BLOCK_SIZE_M, BLOCK_SIZE_N)
        # `acc` tells Triton to write the output of the matmul directly to accumulator, which is more efficient than
        #   accumulator += tl.dot(a, b)

        # advance the pointers to the next block along K
        a_offsets += BLOCK_SIZE_K * stride_a_K  
        b_offsets += BLOCK_SIZE_K * stride_b_K

    # write back the block of the output matrix C with masks
    c_offsets = stride_c_M * offsets_M[:, None] + stride_c_N * offsets_N[None, :]
    c_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N) # notice the 2D mask
    tl.store(c_ptr + c_offsets, tl.exp(accumulator).to(tl.float16), mask=c_mask) # shape (BLOCK_SIZE_M, BLOCK_SIZE_N)


######### Step 2 #########
def rbf1(a, b):
    # check constraints
    assert a.ndim == b.ndim == 2, "only supports matrices, not vectors or tensors"
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    #assert a.is_contiguous() and b.is_contiguous, "input matrices must be contiguous"
    a, b = a.to(torch.float16), b.to(torch.float16)
    (M, K), (_, N) = a.shape, b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
  
    # cdiv(x, y) = (x + (y - 1)) // y
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    _rbf1_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

def rbf2(a, b):
    # check constraints
    assert a.ndim == b.ndim == 2, "only supports matrices, not vectors or tensors"
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    #assert a.is_contiguous() and b.is_contiguous, "input matrices must be contiguous"
    a, b = a.to(torch.float16), b.to(torch.float16)
    
    (M, K), (_, N) = a.shape, b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    _rbf2_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

def rbf_custom(a, b, data):
    # Extract kernel parameters from data dictionary
    BLOCK_SIZE_M = data['BLOCK_SIZE_M']
    BLOCK_SIZE_N = data['BLOCK_SIZE_N'] 
    BLOCK_SIZE_K = data['BLOCK_SIZE_K']
    GROUP_SIZE = data['GROUP_SIZE']
    num_stages = data['num_stages']
    num_warps = data['num_warps']
    # check constraints
    assert a.ndim == b.ndim == 2, "only supports matrices, not vectors or tensors"
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    #assert a.is_contiguous() and b.is_contiguous, "input matrices must be contiguous"
    a, b = a.to(torch.float16), b.to(torch.float16)
    
    (M, K), (_, N) = a.shape, b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    _rbf_custom_kernel[grid](
        a, b, c,
                                    M, N, K,
                                    a.stride(0), a.stride(1),
                                    b.stride(0), b.stride(1),
                                    c.stride(0), c.stride(1),      
                                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                                    GROUP_SIZE=GROUP_SIZE,
                                    num_stages=num_stages,
                                    num_warps=num_warps,)
    return c

def rbf_kernel(x, y):
    """
    Compute the RBF kernel matrix K(x, y) = exp(-gamma * ||x - y||^2)
    
    Args:
        x (torch.Tensor): Tensor of shape (m, d)
        y (torch.Tensor): Tensor of shape (n, d)
        gamma (float): Kernel coefficient (gamma = 1 / (2 * sigma^2))
    
    Returns:
        torch.Tensor: Kernel matrix of shape (m, n)
    """
    x_norm = torch.sum(x**2, dim=1, keepdim=True)  # Shape: (m, 1)
    y_norm = torch.sum(y**2, dim=1, keepdim=True)  # Shape: (n, 1)
    dist_sq = x_norm - 2 * torch.mm(x, y.T) + y_norm.T  # Shape: (m, n)

    return torch.exp(-dist_sq)


######### Step 1 #########
def test_rbf_kernel(size: tuple, atol=1e-2, rtol=1e-1, device=DEVICE): # TODO does rtol=0 mean we don't use rtol?
    """
    Here is where we test the wrapper function and kernel that we wrote 
    above to ensure all our values are correct, using pytorch as the 
    correct answer to compare against

    We use higher tolerance values than previous tests because all the flop 
    accumulation can really compound when it comes to a matmul; even slight
    differences in the block size and launch grid ordering from what PyTorch 
    does can result in pretty sizeable discrepancies
    """
    # create input data
    torch.manual_seed(0)
    assert type(size) == tuple and len(size) == 2
    a = torch.randn((size[0], size[1]), device=DEVICE, dtype=torch.float16)
    b = torch.randn((size[1], size[0]), device=DEVICE, dtype=torch.float16)
    
    # run pytorch reference implementation
    print(f'Testing with size {size}')
    print('Running PyTorch reference...')
    c_ref = rbf_kernel(a, b)
    
    # Test each kernel based on configuration
    for kernel_name in KERNEL_CONFIG['test_kernels']:
        print(f'Testing {kernel_name}...')
        if kernel_name == 'rbf1':
            c_tri = rbf1(a, b)
            torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
            print(f"✅ {kernel_name} PASSED")
        elif kernel_name == 'rbf2':
            c_tri = rbf2(a, b)
            torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
            print(f"✅ {kernel_name} PASSED")
        elif kernel_name == 'rbf_custom':
            # For custom kernel, we need to provide configuration data
            data = {
                'M': size[0], 'N': size[0], 'K': size[1],
                'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,
                'GROUP_SIZE': 8, 'num_stages': 4, 'num_warps': 8,
            }
            c_tri = rbf_custom(a, b, data)
            torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
            print(f"✅ {kernel_name} PASSED")
        else:
            print(f"❌ Unknown kernel: {kernel_name}")
    
    print("All configured kernels passed! 🎉")

def run_all_tests():
    """Run tests for all configured sizes and kernels."""
    print("=" * 50)
    print("RUNNING ALL TESTS")
    print("=" * 50)
    
    for size in KERNEL_CONFIG['test_sizes']:
        print(f"\nTesting size: {size}")
        test_rbf_kernel(size=size)
    
    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED")
    print("=" * 50)

######### Step 4 #########
configs = [
    triton.testing.Benchmark(
        x_names = ["M", "N", "K"], # we can increase multiple dimensions simultaneously while benchmarking
        x_vals = [128 * i for i in range(2, 33)],
        line_arg = "provider", 
        line_vals = ["relu, matmul", "torch", "triton2", "triton_custom"],
        line_names = ["Regular", "PyTorch", "Triton2", "Triton_Custom"],
        styles = [("orange", "-"), ("green", "-"), ("red", "-"), ("pink", "-")],
        ylabel = "Sec", 
        plot_name = "rbf-performance",
        args={},
    )
]

data = None

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.05, 0.95]
    if provider == 'relu, matmul':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.nn.functional.relu(torch.matmul(a, b)), quantiles=quantiles)
    elif provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rbf_kernel(a, b), quantiles=quantiles)
    #if provider == 'triton1':
    #    ms, min_ms, max_ms = triton.testing.do_bench(lambda: rbf1(a, b), quantiles=quantiles)
    elif provider == 'triton2':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rbf2(a, b), quantiles=quantiles)
    elif provider == 'triton_custom':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rbf_custom(a, b, data), quantiles=quantiles)
    else:
        raise ValueError(f"Unknown Provider {provider}")
    perf = lambda ms: ms * 1e-3
    return perf(ms), perf(max_ms), perf(min_ms)

def configs(data):
    M = data['M']
    N = data['N']
    K = data['K']
    BLOCK_SIZE_M = data['BLOCK_SIZE_M']
    BLOCK_SIZE_N = data['BLOCK_SIZE_N']
    BLOCK_SIZE_K = data['BLOCK_SIZE_K']
    GROUP_SIZE = data['GROUP_SIZE']
    num_stages = data['num_stages']
    num_warps = data['num_warps']
    import math
    num_PID_along_M = math.ceil(M / BLOCK_SIZE_M) # the number of blocks along M dimension
    num_PID_along_N = math.ceil(N /BLOCK_SIZE_N) # the number of blocks along N dimension

    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)

    properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
    NUM_SM = properties["multiprocessor_count"] 
    NUM_REGS = properties["max_num_regs"] # number of registers in an SM
    TOTAL_SRAM_PER_SM = properties["max_shared_mem"] 
    WARP_SIZE = properties["warpSize"]
    print(
    f"Number of SMs: {NUM_SM}\n"
    f"Number of registers per SM: {NUM_REGS}\n" 
    f"Total SRAM per SM: {TOTAL_SRAM_PER_SM} bytes\n"
    f"Warp size: {WARP_SIZE} threads"
    )

    kernel = _rbf_custom_kernel.warmup(a, b, c,
                                    M, N, K,
                                    a.stride(0), a.stride(1),
                                    b.stride(0), b.stride(1),
                                    c.stride(0), c.stride(1),      
                                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                                    GROUP_SIZE=GROUP_SIZE,
                                    num_stages=num_stages,
                                    num_warps=num_warps,
                                    grid=(1,))


    kernel._init_handles()
    n_regs = kernel.n_regs # per thread registers usage
    sram_needed_per_program = kernel.metadata.shared # sram_per_program usage
    reg_occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program
    programs_per_sm = min(reg_occupancy, sram_occupancy)
    num_programs_by_limit = NUM_SM * programs_per_sm
    num_programs_by_desire = num_PID_along_M * num_PID_along_N
    print(
        f"Registers per thread: {n_regs}\n"
        f"SRAM per program: {sram_needed_per_program} bytes\n"
        f"Register occupancy: {reg_occupancy} programs per SM\n" 
        f"SRAM occupancy: {sram_occupancy} programs per SM\n"
        f"Programs per SM: {programs_per_sm}\n"
        f"Total programs by hardware limit: {num_programs_by_limit}\n"
        f"Total programs desired: {num_programs_by_desire}"
    )

    print(
        f"Register occupancy percentage: {((n_regs * WARP_SIZE * num_warps) / NUM_REGS) * 100:.1f}%\n" 
        f"SRAM occupancy percentage: {(sram_needed_per_program / (TOTAL_SRAM_PER_SM / programs_per_sm)) * 100:.1f}%\n" 
        f"Pipelining of Programs, current programs num vs simul programs: {(num_programs_by_limit / num_programs_by_desire) * 100:.1f}%"
    )

    for _ in range(1):
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
        _rbf_custom_kernel[grid](
            a, b, c,
                                        M, N, K,
                                        a.stride(0), a.stride(1),
                                        b.stride(0), b.stride(1),
                                        c.stride(0), c.stride(1),      
                                        BLOCK_SIZE_M=BLOCK_SIZE_M,
                                        BLOCK_SIZE_N=BLOCK_SIZE_N,
                                        BLOCK_SIZE_K=BLOCK_SIZE_K,
                                        GROUP_SIZE=GROUP_SIZE,
                                        num_stages=num_stages,
                                        num_warps=num_warps,)




if __name__ == "__main__":
    data = {
            'M': 4096,
            'N': 4096,
            'K': 4096,
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 128,
            'GROUP_SIZE': 8,
            'num_stages': 8,
            'num_warps': 8,
        }
    import sys
    if len(sys.argv) > 0 and sys.argv[1] == "--configs":
        configs(data)
        sys.exit(0)

    # always run unit-tests
    test_rbf_kernel(size=(1024, 1024))

    # Only run benchmark if explicitly requested
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)
    