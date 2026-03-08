
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <stdio.h>
#include <torch/extension.h>

#define PACK_FACTOR 8
#define WARP_SIZE 32

// --- Helper Functions ---
__device__ __forceinline__ float warp_reduce_sum(float sum) {
    #pragma unroll
    for(int i = 4; i >= 0; i--){
        sum += __shfl_down_sync(0xffffffff, sum, 1<<i);
    }
    return sum;
}

__device__ __forceinline__ int make_divisible(int c, int divisor){
    return (c + divisor - 1) / divisor;
}

/*
Computes GEMV (group_size = 64).

Args:
  inputs: vector of shape [batch_size, IC];
  weight: matrix of shape [OC, IC / 8];
  output: vector of shape [OC];
  zeros: matrix of shape [OC, IC / group_size / 8];
  scaling_factors: matrix of shape [OC, IC / group_size];

Notes:
  One cannot infer group_size from the shape of scaling factors.
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
__global__ void gemv_kernel_g64(
  const float4* _inputs, const uint32_t* weight, const uint32_t* zeros, const half* scaling_factors, half* _outputs, 
  const int IC, const int OC){
    const int group_size = 64;
    float psum = 0;
    const int batch_idx = blockIdx.z;
    const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const float4* inputs = _inputs + batch_idx * IC / PACK_FACTOR;
    half* outputs = _outputs + batch_idx * OC;
    // This is essentially zeros_w.
    const int num_groups_packed = make_divisible(make_divisible(IC / group_size, PACK_FACTOR), 2) * 2;
    const int weight_w = IC / PACK_FACTOR;
    // TODO (Haotian): zeros_w is incorrect, after fixing we got misaligned address
    const int zeros_w = make_divisible(make_divisible(IC / group_size, PACK_FACTOR), 2) * 2;
    // consistent with input shape
    const int sf_w = make_divisible(make_divisible(IC / group_size, PACK_FACTOR), 2) * 2 * PACK_FACTOR;
    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("%d %d %d %d %d\n", IC, group_size, PACK_FACTOR, zeros_w, sf_w);
    // tile size: 4 OC x 1024 IC per iter
    for(int packed_group_idx = 0; packed_group_idx < num_groups_packed / 2; packed_group_idx++){
      // 1024 numbers in one iteration across warp. Need 1024 / group_size zeros.
      uint64_t packed_zeros = *reinterpret_cast<const uint64_t*>(zeros + oc_idx * zeros_w + packed_group_idx * 2);
      uint32_t packed_weights[4];
      // use float4 to load weights, each thread load 32 int4 numbers (1 x float4)
      *((float4*)(packed_weights)) = *((float4*)(weight + oc_idx * weight_w + packed_group_idx * (WARP_SIZE * 4) + threadIdx.x * 4));
      // load scaling factors
      // g64: two threads -> 64 numbers -> 1 group; 1 warp = 16 groups.
      float scaling_factor = __half2float(scaling_factors[oc_idx * sf_w + packed_group_idx * 16 + (threadIdx.x / 2)]);
      float current_zeros = (float)((packed_zeros >> (threadIdx.x / 2 * 4)) & 0xF);
      int inputs_ptr_delta = packed_group_idx * WARP_SIZE * 4 + threadIdx.x * 4; 
      const float4* inputs_ptr = inputs + inputs_ptr_delta;
      // multiply 32 weights with 32 inputs
      #pragma unroll
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        // iterate over different uint32_t packed_weights in this loop
        uint32_t current_packed_weight = packed_weights[ic_0];
        half packed_inputs[PACK_FACTOR];
        // each thread load 8 inputs, starting index is packed_group_idx * 128 * 8 (because each iter loads 128*8)
        if (inputs_ptr_delta + ic_0 < IC / PACK_FACTOR) {
          *((float4*)packed_inputs) = *(inputs_ptr + ic_0);
          #pragma unroll
          for (int ic_1 = 0; ic_1 < PACK_FACTOR; ic_1++){
            // iterate over 8 numbers packed within each uint32_t number
            float current_single_weight_fp = (float)(current_packed_weight & 0xF);
            float dequantized_weight = scaling_factor * (current_single_weight_fp - current_zeros);
            //if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && ic_0 == 0 && ic_1 == 0 && packed_group_idx == 0) printf("%f %f %f %f %X %X\n", dequantized_weight, current_single_weight_fp, scaling_factor, current_zeros, current_packed_weight, packed_zeros);
            psum += dequantized_weight * __half2float(packed_inputs[ic_1]);
            current_packed_weight = current_packed_weight >> 4;
          }
        }
      }
    }
    psum = warp_reduce_sum(psum);
    if (threadIdx.x == 0) {
     outputs[oc_idx] = __float2half(psum); 
    }
}


/*
Computes GEMV (group_size = 128).

Args:
  inputs: vector of shape [batch_size, IC];
  weight: matrix of shape [OC, IC / 8];
  output: vector of shape [OC];
  zeros: matrix of shape [OC, IC / group_size / 8];
  scaling_factors: matrix of shape [OC, IC / group_size];

Notes:
  One cannot infer group_size from the shape of scaling factors.
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
__global__ void gemv_kernel_g128(
  // The weight matrix is packed, 8 weights per uint32_t
  // The zeros are also packed, 8 zeros per uint32_t
  // the inputs should be have, FP16, float4 is a vector of 4 FP32, so it each float4 can hold 8 FP16 numbers.
  const float4* _inputs, const uint32_t* weight, const uint32_t* zeros, const half* scaling_factors, half* _outputs, 
  const int IC, const int OC){
    const int group_size = 128;
    float psum = 0;
    const int batch_idx = blockIdx.z;
    const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const float4* inputs = _inputs + batch_idx * IC / PACK_FACTOR; // We divide by PACK_FACTOR because each float4 holds 8 FP16 numbers
    half* outputs = _outputs + batch_idx * OC;
    const int num_groups_packed = make_divisible(IC / group_size, PACK_FACTOR);
    const int weight_w = IC / PACK_FACTOR; 
    // TODO (Haotian): zeros_w is incorrect, after fixing we got misaligned address
    const int zeros_w = make_divisible(IC / group_size, PACK_FACTOR);
    // consistent with input shape
    const int sf_w = make_divisible(IC / group_size, PACK_FACTOR) * PACK_FACTOR;
    //if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("%d %d %d %d\n", IC, group_size, PACK_FACTOR, zeros_w);
    // tile size: 4 OC x 1024 IC per iter

    // we iterate over REAL groups here and when there are no groups more than IF axis is exhausted.
    for(int packed_group_idx = 0; packed_group_idx < num_groups_packed; packed_group_idx++){
      // 1024 numbers in one iteration across warp. Need 1024 / group_size zeros.
      uint32_t packed_zeros = *(zeros + oc_idx * zeros_w + packed_group_idx);
      uint32_t packed_weights[4];
      // we load an array of 4 uint32_t which means we load 8 * 4 = 32 weight at once per thread. 32*32 = 1024 W per warp
      // use float4 to load weights, each thread load 32 int4 numbers (1 x float4)
      *((float4*)(packed_weights)) = *((float4*)(weight + oc_idx * weight_w + packed_group_idx * (WARP_SIZE * 4) + threadIdx.x * 4));
      // w + gets us to correct row + we correct by the number we advance (EACH WARP TAKES 32 times 4-vector) + the offset of the thread in the warp
      // load scaling factors
      // g128: four threads -> 128 numbers -> 1 group; 1 warp = 8 groups.
      float scaling_factor = __half2float(scaling_factors[oc_idx * sf_w + packed_group_idx * 8 + (threadIdx.x / 4)]);
      float current_zeros = (float)((packed_zeros >> (threadIdx.x / 4 * 4)) & 0xF);
      int inputs_ptr_delta = packed_group_idx * WARP_SIZE * 4 + threadIdx.x * 4; 
      const float4* inputs_ptr = inputs + inputs_ptr_delta;
      // multiply 32 weights with 32 inputs
      #pragma unroll
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        // iterate over different uint32_t packed_weights in this loop
        uint32_t current_packed_weight = packed_weights[ic_0];
        half packed_inputs[PACK_FACTOR];
        // each thread load 8 inputs, starting index is packed_group_idx * 128 * 8 (because each iter loads 128*8)
        if (inputs_ptr_delta + ic_0 < IC / PACK_FACTOR) {
          *((float4*)packed_inputs) = *(inputs_ptr + ic_0);
          #pragma unroll
          for (int ic_1 = 0; ic_1 < PACK_FACTOR; ic_1++){
            // iterate over 8 numbers packed within each uint32_t number
            float current_single_weight_fp = (float)(current_packed_weight & 0xF);
            float dequantized_weight = scaling_factor * (current_single_weight_fp - current_zeros);
            //if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && ic_0 == 0 && ic_1 == 0 && packed_group_idx == 0) printf("%f %f %f %f %X %X\n", dequantized_weight, current_single_weight_fp, scaling_factor, current_zeros, current_packed_weight, packed_zeros);
            psum += dequantized_weight * __half2float(packed_inputs[ic_1]);
            current_packed_weight = current_packed_weight >> 4;
          }
        }
      }
    }
    psum = warp_reduce_sum(psum);
    if (threadIdx.x == 0) {
     outputs[oc_idx] = __float2half(psum); 
    }
}

// --- The C++ Wrapper ---
torch::Tensor awq_forward(
    torch::Tensor _inputs, 
    torch::Tensor weight, 
    torch::Tensor zeros, 
    torch::Tensor scaling_factors, 
    int IC, int OC, int group_size) 
{
    // Basic validation
    TORCH_CHECK(_inputs.is_cuda(), "inputs must be on CUDA");
    
    // Ensure input is half (FP16)
    auto inputs_f16 = _inputs.to(torch::kHalf);
    
    // Allocate output: [Batch, OC]
    auto out_feats = torch::empty({_inputs.size(0), OC}, 
                                 torch::TensorOptions().dtype(torch::kHalf).device(_inputs.device()));

    // AWQ kernel grid
    // blockDim.y=4 means it processes 4 rows (OC) per block
    // IC is x dim, OC is y dim and batch size is z dim because we want to process different samples in parallel
    dim3 num_blocks(1, OC / 4, _inputs.size(0));
    // one warp per IC channels but 4 across the OC. 4 total
    dim3 num_threads(32, 4);

    if (group_size == 64) {
        gemv_kernel_g64<<<num_blocks, num_threads>>>(
            (const float4*)inputs_f16.data_ptr<at::Half>(),
            (const uint32_t*)weight.data_ptr<uint8_t>(),
            (const uint32_t*)zeros.data_ptr<uint8_t>(),
            (const half*)scaling_factors.data_ptr<at::Half>(),
            (half*)out_feats.data_ptr<at::Half>(),
            IC, OC);
    } else {
        gemv_kernel_g128<<<num_blocks, num_threads>>>(
            (const float4*)inputs_f16.data_ptr<at::Half>(),
            (const uint32_t*)weight.data_ptr<uint8_t>(),
            (const uint32_t*)zeros.data_ptr<uint8_t>(),
            (const half*)scaling_factors.data_ptr<at::Half>(),
            (half*)out_feats.data_ptr<at::Half>(),
            IC, OC);
    }
    
    return out_feats;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &awq_forward, "AWQ forward");
}