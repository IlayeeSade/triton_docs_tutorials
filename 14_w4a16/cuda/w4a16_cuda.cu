#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------
// 1. The Optimized GEMV Kernel (B=1) using Parallel Reduction
// ---------------------------------------------------------
__global__ void w4a16_gemv_optimized_kernel(
    const uint8_t* __restrict__ W_packed, 
    const half* __restrict__ b,           
    const half* __restrict__ S,           
    const half* __restrict__ Z,           
    const half* __restrict__ activations, 
    half* __restrict__ OUT,               
    int OF, int IF, int group_size) 
{
    // 1 Block computes exactly 1 Output Feature (Row)
    int current_of = blockIdx.x; 
    
    // Thread ID within this block (0 to 255)
    int tid = threadIdx.x;

    // Figure out which packed row we are reading from
    int OF_packed = OF / 2;
    int pid_half = current_of / OF_packed;          
    int current_of_packed = current_of % OF_packed; 

    // Each thread holds its own partial sum in ultra-fast registers
    float partial_acc = 0.0f;

    // STEP 1: Coalesced Grid-Stride Loop
    for (int k = tid; k < IF; k += blockDim.x) {
        
        // Read the packed byte
        uint8_t w_packed_val = W_packed[current_of_packed * IF + k];
        
        // Unpack based on which half of the output feature matrix we are in
        uint8_t w_unpacked = (pid_half == 0) ? (w_packed_val & 0x0F) : ((w_packed_val >> 4) & 0x0F);

        // Fetch scale and zero point
        int group_idx = k / group_size;
        float s_val = __half2float(S[current_of * (IF / group_size) + group_idx]);
        float z_val = __half2float(Z[current_of * (IF / group_size) + group_idx]);

        // Dequantize the weight
        float w_deq = (static_cast<float>(w_unpacked) - z_val) * s_val;
        
        // Read activation and accumulate
        float act_val = __half2float(activations[k]);
        
        partial_acc += w_deq * act_val;
    }

    // STEP 2: Warp-Level Reduction
    // A warp is a group of 32 threads. They can share data directly without RAM.
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_acc += __shfl_down_sync(0xffffffff, partial_acc, offset);
    }

    // STEP 3: Block-Level Reduction via Shared Memory
    // We have 256 threads = 8 warps. We need to sum the 8 results from the warps.
    __shared__ float shared_acc[32]; 
    
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // The first thread in each warp writes its result to shared memory
    if (lane_id == 0) {
        shared_acc[warp_id] = partial_acc;
    }
    
    // Ensure all warps have written to shared memory before continuing
    __syncthreads();

    // STEP 4: Final sum by the very first warp
    if (warp_id == 0) {
        // Read the 8 warp results (other threads read 0.0)
        partial_acc = (lane_id < (blockDim.x / 32)) ? shared_acc[lane_id] : 0.0f;
        
        // Do one last reduction to get the final total in thread 0
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_acc += __shfl_down_sync(0xffffffff, partial_acc, offset);
        }
    }

    // STEP 5: Thread 0 adds the bias and writes to Global Memory
    if (tid == 0) {
        float bias_val = __half2float(b[current_of]);
        OUT[current_of] = __float2half(partial_acc + bias_val);
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

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(W_packed.device());
    auto OUT = torch::empty({OF, B}, options);

    // Grid Setup: 1 Block per Output Feature. 256 Threads per Block.
    int threads = 256; 
    int blocks = OF;

    w4a16_gemv_optimized_kernel<<<blocks, threads>>>(
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