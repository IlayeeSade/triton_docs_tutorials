#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BLOCK_DIM_X 128
#define BLOCK_DIM_Y 2

// ---------------------------------------------------------
// 1. The VECTORIZED Shared Weight Broadcast Kernel
// ---------------------------------------------------------

__global__ void w4a16_gemv_vectorized_kernel(
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

    const uint32_t* W_vec = reinterpret_cast<const uint32_t*>(W_packed);
    int IF_vec = IF / 4; 

    // Main data shared memory
    __shared__ float shm_act[BLOCK_DIM_X * 4];

    float partial_acc = 0.0f;

    for (int k_base = 0; k_base < IF_vec; k_base += BLOCK_DIM_X) {
        int k_vec = k_base + tx;

        // Fetch 4 activations
        if (ty == 0 && k_vec < IF_vec) {
            int act_idx = k_vec * 4;
            shm_act[tx * 4 + 0] = __half2float(__ldg(&activations[act_idx + 0]));
            shm_act[tx * 4 + 1] = __half2float(__ldg(&activations[act_idx + 1]));
            shm_act[tx * 4 + 2] = __half2float(__ldg(&activations[act_idx + 2]));
            shm_act[tx * 4 + 3] = __half2float(__ldg(&activations[act_idx + 3]));
        }
        __syncthreads();

        if (k_vec < IF_vec) {
            uint32_t w_chunk = W_vec[current_of_packed * IF_vec + k_vec];

            int group_idx = (k_vec * 4) / group_size / 8; // Calculate group index for scaling factors and zeros
            float s_val = __half2float(__ldg(&S[row_idx * (IF / group_size) + group_idx]));
            // uint32_t z_intermediate_ptrs = *(Z + row_idx * (IF / group_size / 8) + group_idx);
            float z_val = __half2float(__ldg(&Z[row_idx * (IF / group_size) + group_idx]));

            #pragma unroll
            for(int step = 0; step < 4; step++) {
                uint8_t w_packed_val = (w_chunk >> (step * 8)) & 0xFF;
                uint8_t w_unpacked = (ty == 0) ? (w_packed_val & 0x0F) : ((w_packed_val >> 4) & 0x0F);

                float w_deq = (static_cast<float>(w_unpacked) - z_val) * s_val;
                float act_val = shm_act[tx * 4 + step];

                partial_acc += w_deq * act_val;
            }
        }
        __syncthreads();
    }

    // --- REDUCTION STAGE ---
    
    // 1. Warp-level reduction (Intra-warp)
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_acc += __shfl_down_sync(0xffffffff, partial_acc, offset);
    }

    // 2. Shared memory for Cross-warp reduction
    // Size is dynamically calculated based on warps per block
    constexpr int NUM_WARPS = BLOCK_DIM_X / 32;
    __shared__ float shm_reduce[BLOCK_DIM_Y][NUM_WARPS];
    
    int warp_id = tx / 32;
    int lane_id = tx % 32;
    
    if (lane_id == 0) {
        shm_reduce[ty][warp_id] = partial_acc;
    }
    __syncthreads();

    // 3. Final Block-level reduction (Inter-warp)
    if (tx == 0) {
        float final_acc = 0.0f;
        
        #pragma unroll
        for (int i = 0; i < NUM_WARPS; i++) {
            final_acc += shm_reduce[ty][i];
        }

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