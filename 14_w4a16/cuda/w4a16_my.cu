#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 4

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
    int current_of_packed = blockIdx.x * BLOCK_DIM_Y + ty; 
    int row_idx = current_of_packed * 2;
    if (current_of_packed >= OF / 2) return;

    const uint32_t* W_vec = reinterpret_cast<const uint32_t*>(W_packed);
    int IF_vec = IF / 4; 

    float partial_acc_0 = 0.0f;
    float partial_acc_1 = 0.0f;

    for (int k_base = 0; k_base < IF_vec; k_base += BLOCK_DIM_X) {
        int k_vec = k_base + tx;

        if (k_vec < IF_vec) {
            // Fetch 4 activations directly into registers. 
            // The hardware L1 cache will automatically broadcast to threads sharing the same act_idx.
            int act_idx = k_vec * 4;
            float local_act[4];

            // ldg because we will never write to these activations.
            uint64_t packed_acts = __ldg(reinterpret_cast<const uint64_t*>(&activations[act_idx]));
            const half* h = reinterpret_cast<const half*>(&packed_acts);

            local_act[0] = __half2float(h[0]);
            local_act[1] = __half2float(h[1]);
            local_act[2] = __half2float(h[2]);
            local_act[3] = __half2float(h[3]);
            
            // We load 32  4
            uint32_t w_chunk = W_vec[current_of_packed * IF_vec + k_vec];

            int group_idx = (k_vec * 4) / group_size; // Calculate group index for scaling factors and zeros

            // grabbing 16 bit two byte float, even though the bus is 128 bit.
            // : float s_val = __half2float(__ldg(&S[row_idx * (IF / group_size) + group_idx])) :
            // notice we need to grab 16 bits for every activation so ideally we want 128 / 16 = 8 groups per thread
            // Maybe we could use some other compression which prepares scaling and zeros for the next turns
            // TODO 
            float s_val_0 = __half2float(__ldg(&S[row_idx * (IF / group_size) + group_idx]));
            float s_val_1 = __half2float(__ldg(&S[(row_idx + 1) * (IF / group_size) + group_idx]));
            // uint32_t z_intermediate_ptrs = *(Z + row_idx * (IF / group_size / 8) + group_idx);
            float z_val_0 = __half2float(__ldg(&Z[row_idx * (IF / group_size) + group_idx]));
            float z_val_1 = __half2float(__ldg(&Z[(row_idx + 1) * (IF / group_size) + group_idx]));

            #pragma unroll
            for(int step = 0; step < 4; step++) {
                uint8_t w_packed_val = (w_chunk >> (step * 8)) & 0xFF;
                uint8_t w_up_0 = w_packed_val & 0x0F;
                uint8_t w_up_1 = (w_packed_val >> 4) & 0x0F;

                float w_deq_0 = (static_cast<float>(w_up_0) - z_val_0) * s_val_0;
                float w_deq_1 = (static_cast<float>(w_up_1) - z_val_1) * s_val_1;
                float act_val = local_act[step];

                partial_acc_0 += w_deq_0 * act_val;
                partial_acc_1 += w_deq_1 * act_val;
            }
        }
    }

    // --- REDUCTION STAGE ---
    
    // 1. Warp-level reduction (Intra-warp)
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_acc_0 += __shfl_down_sync(0xffffffff, partial_acc_0, offset);
        partial_acc_1 += __shfl_down_sync(0xffffffff, partial_acc_1, offset);
    }

    // *** cross warp reduction would go here if we had more than 32 threads per block *** //
    // // 2. Shared memory for Cross-warp reduction
    // // Size is dynamically calculated based on warps per block
    // constexpr int NUM_WARPS = BLOCK_DIM_X / 32;
    // __shared__ float shm_reduce[BLOCK_DIM_Y][NUM_WARPS];
    
    // int warp_id = tx / 32;
    // int lane_id = tx % 32;
    
    // if (lane_id == 0) {
    //     shm_reduce[ty][warp_id] = partial_acc;
    // }
    // __syncthreads();

    // 3. Final Block-level reduction (Inter-warp)
    if (tx == 0) {

        // ** cross warp ** //
        // float partial_acc = 0.0f;
        
        // #pragma unroll
        // for (int i = 0; i < NUM_WARPS; i++) {
        //     partial_acc += shm_reduce[ty][i];
        // }

        // Merging
        // float bias_val_0 = __half2float(b[row_idx]);
        // float bias_val_1 = __half2float(b[row_idx + 1]);

        // OUT[row_idx] = __float2half(partial_acc_0 + bias_val_0);
        // OUT[row_idx+1] = __float2half(partial_acc_1 + bias_val_1);

        half2 bias_h2 = __ldg(reinterpret_cast<const half2*>(&b[row_idx]));
        float2 bias_f2 = __half22float2(bias_h2);

        half2 out_h2 = __floats2half2_rn(
            partial_acc_0 + bias_f2.x,
            partial_acc_1 + bias_f2.y
        );

        *reinterpret_cast<half2*>(&OUT[row_idx]) = out_h2;
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
    dim3 blocks((OF_packed + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

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