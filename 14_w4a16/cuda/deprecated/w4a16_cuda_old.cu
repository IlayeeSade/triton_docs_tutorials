#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BLOCK_DIM_X 64
#define BLOCK_DIM_Y 2

// ---------------------------------------------------------
// 1. The VECTORIZED Shared Weight Broadcast Kernel
// ---------------------------------------------------------

// ---------------------------------------------------------
// 1. The uint4 VECTORIZED Kernel (Maximum Bandwidth)
// ---------------------------------------------------------
__global__ void w4a16_gemv_uint4_kernel(
    const uint8_t* __restrict__ W_packed, 
    const half* __restrict__ b,           
    const half* __restrict__ S,           
    const half* __restrict__ Z,           
    const half* __restrict__ activations, 
    half* __restrict__ OUT,               
    int OF, int IF, int group_size) 
{
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int current_of_packed = blockIdx.x;
    int row_idx = (current_of_packed * 2) + ty;

    // THE UPGRADE: 128-bit memory transactions! (16 bytes at once)
    // uses a 4-vector of uint32_t, but it now uses 1 instruction for looping while
    // the previous things needed 4 instructions for looping
    // and optimizes syncthreads, less total calls
    const uint4* W_vec = reinterpret_cast<const uint4*>(W_packed);
    
    // We loop 1/16th as many times now
    int IF_vec = IF / 16; 

    // 16 activations per thread
    __shared__ float shm_act[BLOCK_DIM_X * 16];

    float partial_acc = 0.0f;

    for (int k_base = 0; k_base < IF_vec; k_base += BLOCK_DIM_X) {
        int k_vec = k_base + tx;
        int base_k = k_vec * 16; // The actual starting column for this chunk

        // ty=0 fetches 16 activations into shared memory
        if (ty == 0 && k_vec < IF_vec) {
            #pragma unroll
            for(int i = 0; i < 16; i++) {
                shm_act[tx * 16 + i] = __half2float(__ldg(&activations[base_k + i]));
            }
        }
        __syncthreads();

        if (k_vec < IF_vec) {
            // THE GULP: Fetch 128-bits (16 weights) in a single hardware instruction
            uint4 w_chunk_128 = W_vec[current_of_packed * IF_vec + k_vec];
            
            // Break it down into four 32-bit registers for easy processing
            uint32_t chunks[4] = {w_chunk_128.x, w_chunk_128.y, w_chunk_128.z, w_chunk_128.w};

            // ==========================================================
            // THE JACKPOT: Because we load 16 weights, and group_size is 16,
            // this entire chunk shares exactly ONE Scale and ONE Zero-point!
            // ==========================================================
            int group_idx = base_k / group_size; 
            float s_val = __half2float(__ldg(&S[row_idx * (IF / group_size) + group_idx]));
            float z_val = __half2float(__ldg(&Z[row_idx * (IF / group_size) + group_idx]));

            // Process all 16 weights in pure, blindingly fast register math
            #pragma unroll
            for(int c = 0; c < 4; c++) {
                #pragma unroll
                for(int step = 0; step < 4; step++) {
                    uint8_t w_packed_val = (chunks[c] >> (step * 8)) & 0xFF;
                    uint8_t w_unpacked = (ty == 0) ? (w_packed_val & 0x0F) : ((w_packed_val >> 4) & 0x0F);

                    float w_deq = (static_cast<float>(w_unpacked) - z_val) * s_val;
                    float act_val = shm_act[tx * 16 + (c * 4) + step];

                    partial_acc += w_deq * act_val;
                }
            }
        }
        __syncthreads();
    }

    // Warp Reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_acc += __shfl_down_sync(0xffffffff, partial_acc, offset);
    }

    // Block Reduction
    __shared__ float shm_reduce[BLOCK_DIM_Y][2];
    int warp_id = tx / 32;
    int lane_id = tx % 32;

    if (lane_id == 0) {
        shm_reduce[ty][warp_id] = partial_acc;
    }
    __syncthreads();

    // Output
    if (tx == 0) {
        float final_acc = shm_reduce[ty][0] + shm_reduce[ty][1];
        float bias_val = __half2float(b[row_idx]);
        OUT[row_idx] = __float2half(final_acc + bias_val);
    }
}

__global__ void w4a16_gemv_vectorized_kernel(
    // restric says it does not overlap with nothing 
    const uint8_t* __restrict__ W_packed, 
    const half* __restrict__ b,
    const half* __restrict__ S,           
    const half* __restrict__ Z,           
    const half* __restrict__ activations, 
    half* __restrict__ OUT,               
    int OF, int IF, int group_size) 
{
    // The thread from the thread block
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int current_of_packed = blockIdx.x;
    int row_idx = (current_of_packed * 2) + ty;

    // we cast to 32bits in order to read more at once
    const uint32_t* W_vec = reinterpret_cast<const uint32_t*>(W_packed);
    
    // we only need to loop 1/4 times now
    int IF_vec = IF / 4; 

    // we need 4 the shared memory
    // shared is for all threads in the same thread block
    __shared__ float shm_act[BLOCK_DIM_X * 4];

    float partial_acc = 0.0f;

    // Leap froging
    for (int k_base = 0; k_base < IF_vec; k_base += BLOCK_DIM_X) {
        int k_vec = k_base + tx;

        // Fetch 4 activations sequentially and drop them in shared memory
        if (ty == 0 && k_vec < IF_vec) {
            int act_idx = k_vec * 4;
            // __ldg forces the read through the Read-Only Texture Cache
            // if we had step X tx a 2d arr shm_act then
            // notice that now we read 4 activations at once, because every bank in SRAM is working since we iterate in the fastest changing dimension (the column dimension)
            // and bank are assigned by indices. So the threads work in parrelell

            // after testing we recall that later we unroll the loop in below, then because the channel step in this 1d version is the fastest changing
            // dimension, then we can read 4 activations at once in the same instruction, because they are contiguous in so thats probably
            // the cause for why the latter method is superior.
            shm_act[tx * 4 + 0] = __half2float(__ldg(&activations[act_idx + 0]));
            shm_act[tx * 4 + 1] = __half2float(__ldg(&activations[act_idx + 1]));
            shm_act[tx * 4 + 2] = __half2float(__ldg(&activations[act_idx + 2]));
            shm_act[tx * 4 + 3] = __half2float(__ldg(&activations[act_idx + 3]));
        }
        __syncthreads();

        if (k_vec < IF_vec) {
            uint32_t w_chunk = W_vec[current_of_packed * IF_vec + k_vec];

            // unroll the loop in the compiler to process the 4 bytes instantly in registers
            #pragma unroll

            // We do the slow division once for the whole 4-byte chunk
            // this division outside it crucial
            int group_idx = (k_vec * 4) / group_size;
            float s_val = __half2float(__ldg(&S[row_idx * (IF / group_size) + group_idx]));
            float z_val = __half2float(__ldg(&Z[row_idx * (IF / group_size) + group_idx]));

            for(int step = 0; step < 4; step++) {
                
                // grab the specific 8-bit byte we need right now
                uint8_t w_packed_val = (w_chunk >> (step * 8)) & 0xFF;
                // unpacking logic
                uint8_t w_unpacked = (ty == 0) ? (w_packed_val & 0x0F) : ((w_packed_val >> 4) & 0x0F);

                // int actual_k = k_vec * 4 + step;
                // int group_idx = actual_k / group_size;

                float w_deq = (static_cast<float>(w_unpacked) - z_val) * s_val;
                float act_val = shm_act[tx * 4 + step];

                partial_acc += w_deq * act_val;
            }
        }
        // if one wrap over the same tx range is much faster than a wrap
        // we the same tx range but with a different ty then it will read
        // overwritten stuff
        __syncthreads();
    }

    // all-reduce across a warp, 32 thread 
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_acc += __shfl_down_sync(0xffffffff, partial_acc, offset);
    }

    // block reduction, we have two warps over the x-axis 64/32=2
    __shared__ float shm_reduce[BLOCK_DIM_Y][2];
    
    int warp_id = tx / 32;
    int lane_id = tx % 32;
    
    // we take the captian
    if (lane_id == 0) {
        shm_reduce[ty][warp_id] = partial_acc;
    }
    __syncthreads();

    // captian
    if (tx == 0) {
        float final_acc = shm_reduce[ty][0] + shm_reduce[ty][1];
        float bias_val = __half2float(b[row_idx]);
        OUT[row_idx] = __float2half(final_acc + bias_val);
    }
}

// ---------------------------------------------------------
// 2. The PyTorch C++ Wrapper
// ---------------------------------------------------------
torch::Tensor w4a16_forward(
    torch::Tensor W_packed, 
    torch::Tensor b, 
    torch::Tensor S, 
    torch::Tensor Z, 
    torch::Tensor activations, 
    int group_size) 
{
    TORCH_CHECK(W_packed.is_cuda() && W_packed.is_contiguous(), "W_packed must be CUDA & contiguous");
    TORCH_CHECK(activations.is_cuda() && activations.is_contiguous(), "activations must be CUDA & contiguous");
    
    int OF_packed = W_packed.size(0);
    int IF = W_packed.size(1);
    int B = activations.size(1);
    int OF = OF_packed * 2;
    
    TORCH_CHECK(B == 1, "This optimized kernel strictly requires Batch Size B=1");
    
    // we divide by 4
    TORCH_CHECK(IF % 4 == 0, "Input Features must be a multiple of 4 for 32-bit vectorization");

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(W_packed.device());
    auto OUT = torch::empty({OF, B}, options);

    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y); 
    dim3 blocks(OF_packed);

    w4a16_gemv_vectorized_kernel<<<blocks, threads>>>(
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(S.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Z.data_ptr<at::Half>()),
        reinterpret_cast<half*>(activations.data_ptr<at::Half>()),
        reinterpret_cast<half*>(OUT.data_ptr<at::Half>()),
        OF, IF, group_size
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return OUT;
}

PYBIND11_MODULE(w4a16_cuda_ext, m) {
    m.def("forward", &w4a16_forward, "W4A16 GEMV forward pass (CUDA)");
}