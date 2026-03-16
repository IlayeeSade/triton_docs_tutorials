#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 4

// ---------------------------------------------------------
// 1. The VECTORIZED Shared Weight Broadcast Kernel
//    Each threadblock-y lane now handles 4 packed rows = 8 output rows
// ---------------------------------------------------------

__global__ void w4a16_gemv_vectorized_kernel(
    const uint8_t* __restrict__ W_packed,
    const half* __restrict__ b,
    const half* __restrict__ SZ,
    const half* __restrict__ activations,
    half* __restrict__ OUT,
    int OF, int IF, int group_size)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Each ty now processes 4 packed rows = 8 output rows
    int packed_quad_idx = blockIdx.x * BLOCK_DIM_Y + ty;

    int current_of_packed_0 = packed_quad_idx * 4 + 0;
    int current_of_packed_1 = packed_quad_idx * 4 + 1;
    int current_of_packed_2 = packed_quad_idx * 4 + 2;
    int current_of_packed_3 = packed_quad_idx * 4 + 3;

    int row_idx = current_of_packed_0 * 2; // first of 8 output rows
    const int OF_packed = OF / 2;
    if (current_of_packed_0 >= OF_packed) return;

    bool has_packed_1 = (current_of_packed_1 < OF_packed);
    bool has_packed_2 = (current_of_packed_2 < OF_packed);
    bool has_packed_3 = (current_of_packed_3 < OF_packed);

    const uint32_t* W_vec = reinterpret_cast<const uint32_t*>(W_packed);
    int IF_vec = IF / 4;

    float partial_acc_0 = 0.0f; // row_idx + 0
    float partial_acc_1 = 0.0f; // row_idx + 1
    float partial_acc_2 = 0.0f; // row_idx + 2
    float partial_acc_3 = 0.0f; // row_idx + 3
    float partial_acc_4 = 0.0f; // row_idx + 4
    float partial_acc_5 = 0.0f; // row_idx + 5
    float partial_acc_6 = 0.0f; // row_idx + 6
    float partial_acc_7 = 0.0f; // row_idx + 7

    for (int k_base = 0; k_base < IF_vec; k_base += BLOCK_DIM_X) {
        int k_vec = k_base + tx;

        if (k_vec < IF_vec) {
            // Load 4 activations
            int act_idx = k_vec * 4;

            uint64_t packed_acts = __ldg(reinterpret_cast<const uint64_t*>(&activations[act_idx]));
            const half* h = reinterpret_cast<const half*>(&packed_acts);

            float act0 = __half2float(h[0]);
            float act1 = __half2float(h[1]);
            float act2 = __half2float(h[2]);
            float act3 = __half2float(h[3]);

            // Load up to 4 packed rows
            uint32_t w_chunk_0 = W_vec[current_of_packed_0 * IF_vec + k_vec];
            uint32_t w_chunk_1 = 0;
            uint32_t w_chunk_2 = 0;
            uint32_t w_chunk_3 = 0;

            if (has_packed_1) w_chunk_1 = W_vec[current_of_packed_1 * IF_vec + k_vec];
            if (has_packed_2) w_chunk_2 = W_vec[current_of_packed_2 * IF_vec + k_vec];
            if (has_packed_3) w_chunk_3 = W_vec[current_of_packed_3 * IF_vec + k_vec];

            int group_idx = (k_vec * 4) / group_size;

            // Load [s0,z0,s1,z1,...,s7,z7] for 8 rows
            int base_half = group_idx * (2 * OF) + 2 * row_idx;

            const uint4 packed_sz_0 = reinterpret_cast<const uint4*>(SZ)[base_half >> 3];
            const uint4 packed_sz_1 = reinterpret_cast<const uint4*>(SZ)[(base_half >> 3) + 1];

            const half* vals0 = reinterpret_cast<const half*>(&packed_sz_0);
            const half* vals1 = reinterpret_cast<const half*>(&packed_sz_1);

            float s_val_0 = __half2float(vals0[0]);
            float z_val_0 = __half2float(vals0[1]);
            float s_val_1 = __half2float(vals0[2]);
            float z_val_1 = __half2float(vals0[3]);
            float s_val_2 = __half2float(vals0[4]);
            float z_val_2 = __half2float(vals0[5]);
            float s_val_3 = __half2float(vals0[6]);
            float z_val_3 = __half2float(vals0[7]);

            float s_val_4 = __half2float(vals1[0]);
            float z_val_4 = __half2float(vals1[1]);
            float s_val_5 = __half2float(vals1[2]);
            float z_val_5 = __half2float(vals1[3]);
            float s_val_6 = __half2float(vals1[4]);
            float z_val_6 = __half2float(vals1[5]);
            float s_val_7 = __half2float(vals1[6]);
            float z_val_7 = __half2float(vals1[7]);

            #pragma unroll
            for (int step = 0; step < 4; step++) {
                float act_val = (step == 0 ? act0 : step == 1 ? act1 : step == 2 ? act2 : act3);

                // Packed row 0 -> rows row_idx+0, row_idx+1
                uint8_t w_packed_val_0 = (w_chunk_0 >> (step * 8)) & 0xFF;
                uint8_t w0_lo =  w_packed_val_0       & 0x0F;
                uint8_t w0_hi = (w_packed_val_0 >> 4) & 0x0F;

                partial_acc_0 += (static_cast<float>(w0_lo) - z_val_0) * s_val_0 * act_val;
                partial_acc_1 += (static_cast<float>(w0_hi) - z_val_1) * s_val_1 * act_val;

                // Packed row 1 -> rows row_idx+2, row_idx+3
                if (has_packed_1) {
                    uint8_t w_packed_val_1 = (w_chunk_1 >> (step * 8)) & 0xFF;
                    uint8_t w1_lo =  w_packed_val_1       & 0x0F;
                    uint8_t w1_hi = (w_packed_val_1 >> 4) & 0x0F;

                    partial_acc_2 += (static_cast<float>(w1_lo) - z_val_2) * s_val_2 * act_val;
                    partial_acc_3 += (static_cast<float>(w1_hi) - z_val_3) * s_val_3 * act_val;
                }

                // Packed row 2 -> rows row_idx+4, row_idx+5
                if (has_packed_2) {
                    uint8_t w_packed_val_2 = (w_chunk_2 >> (step * 8)) & 0xFF;
                    uint8_t w2_lo =  w_packed_val_2       & 0x0F;
                    uint8_t w2_hi = (w_packed_val_2 >> 4) & 0x0F;

                    partial_acc_4 += (static_cast<float>(w2_lo) - z_val_4) * s_val_4 * act_val;
                    partial_acc_5 += (static_cast<float>(w2_hi) - z_val_5) * s_val_5 * act_val;
                }

                // Packed row 3 -> rows row_idx+6, row_idx+7
                if (has_packed_3) {
                    uint8_t w_packed_val_3 = (w_chunk_3 >> (step * 8)) & 0xFF;
                    uint8_t w3_lo =  w_packed_val_3       & 0x0F;
                    uint8_t w3_hi = (w_packed_val_3 >> 4) & 0x0F;

                    partial_acc_6 += (static_cast<float>(w3_lo) - z_val_6) * s_val_6 * act_val;
                    partial_acc_7 += (static_cast<float>(w3_hi) - z_val_7) * s_val_7 * act_val;
                }
            }
        }
    }

    // --- REDUCTION STAGE ---

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_acc_0 += __shfl_down_sync(0xffffffff, partial_acc_0, offset);
        partial_acc_1 += __shfl_down_sync(0xffffffff, partial_acc_1, offset);
        partial_acc_2 += __shfl_down_sync(0xffffffff, partial_acc_2, offset);
        partial_acc_3 += __shfl_down_sync(0xffffffff, partial_acc_3, offset);
        partial_acc_4 += __shfl_down_sync(0xffffffff, partial_acc_4, offset);
        partial_acc_5 += __shfl_down_sync(0xffffffff, partial_acc_5, offset);
        partial_acc_6 += __shfl_down_sync(0xffffffff, partial_acc_6, offset);
        partial_acc_7 += __shfl_down_sync(0xffffffff, partial_acc_7, offset);
    }

    constexpr int NUM_WARPS = BLOCK_DIM_X / 32;
    __shared__ float shm_reduce[BLOCK_DIM_Y][NUM_WARPS][8];

    int warp_id = tx / 32;
    int lane_id = tx % 32;

    if (lane_id == 0) {
        shm_reduce[ty][warp_id][0] = partial_acc_0;
        shm_reduce[ty][warp_id][1] = partial_acc_1;
        shm_reduce[ty][warp_id][2] = partial_acc_2;
        shm_reduce[ty][warp_id][3] = partial_acc_3;
        shm_reduce[ty][warp_id][4] = partial_acc_4;
        shm_reduce[ty][warp_id][5] = partial_acc_5;
        shm_reduce[ty][warp_id][6] = partial_acc_6;
        shm_reduce[ty][warp_id][7] = partial_acc_7;
    }
    __syncthreads();

    if (tx == 0) {
        partial_acc_0 = 0.0f;
        partial_acc_1 = 0.0f;
        partial_acc_2 = 0.0f;
        partial_acc_3 = 0.0f;
        partial_acc_4 = 0.0f;
        partial_acc_5 = 0.0f;
        partial_acc_6 = 0.0f;
        partial_acc_7 = 0.0f;

        #pragma unroll
        for (int i = 0; i < NUM_WARPS; i++) {
            partial_acc_0 += shm_reduce[ty][i][0];
            partial_acc_1 += shm_reduce[ty][i][1];
            partial_acc_2 += shm_reduce[ty][i][2];
            partial_acc_3 += shm_reduce[ty][i][3];
            partial_acc_4 += shm_reduce[ty][i][4];
            partial_acc_5 += shm_reduce[ty][i][5];
            partial_acc_6 += shm_reduce[ty][i][6];
            partial_acc_7 += shm_reduce[ty][i][7];
        }

        // Store rows row_idx+0, row_idx+1
        {
            half2 bias_h2 = __ldg(reinterpret_cast<const half2*>(&b[row_idx]));
            float2 bias_f2 = __half22float2(bias_h2);

            half2 out_h2 = __floats2half2_rn(
                partial_acc_0 + bias_f2.x,
                partial_acc_1 + bias_f2.y
            );
            *reinterpret_cast<half2*>(&OUT[row_idx]) = out_h2;
        }

        // Store rows row_idx+2, row_idx+3
        if (has_packed_1) {
            half2 bias_h2 = __ldg(reinterpret_cast<const half2*>(&b[row_idx + 2]));
            float2 bias_f2 = __half22float2(bias_h2);

            half2 out_h2 = __floats2half2_rn(
                partial_acc_2 + bias_f2.x,
                partial_acc_3 + bias_f2.y
            );
            *reinterpret_cast<half2*>(&OUT[row_idx + 2]) = out_h2;
        }

        // Store rows row_idx+4, row_idx+5
        if (has_packed_2) {
            half2 bias_h2 = __ldg(reinterpret_cast<const half2*>(&b[row_idx + 4]));
            float2 bias_f2 = __half22float2(bias_h2);

            half2 out_h2 = __floats2half2_rn(
                partial_acc_4 + bias_f2.x,
                partial_acc_5 + bias_f2.y
            );
            *reinterpret_cast<half2*>(&OUT[row_idx + 4]) = out_h2;
        }

        // Store rows row_idx+6, row_idx+7
        if (has_packed_3) {
            half2 bias_h2 = __ldg(reinterpret_cast<const half2*>(&b[row_idx + 6]));
            float2 bias_f2 = __half22float2(bias_h2);

            half2 out_h2 = __floats2half2_rn(
                partial_acc_6 + bias_f2.x,
                partial_acc_7 + bias_f2.y
            );
            *reinterpret_cast<half2*>(&OUT[row_idx + 6]) = out_h2;
        }
    }
}

// ---------------------------------------------------------
// 2. The PyTorch C++ Wrapper
// ---------------------------------------------------------

torch::Tensor w4a16_forward(
    torch::Tensor W_packed,
    torch::Tensor b,
    torch::Tensor SZ,
    torch::Tensor activations,
    int group_size)
{
    TORCH_CHECK(W_packed.is_cuda(), "W_packed must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(SZ.is_cuda(), "SZ must be a CUDA tensor");
    TORCH_CHECK(activations.is_cuda(), "activations must be a CUDA tensor");

    TORCH_CHECK(W_packed.is_contiguous(), "W_packed must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(SZ.is_contiguous(), "SZ must be contiguous");
    TORCH_CHECK(activations.is_contiguous(), "activations must be contiguous");

    TORCH_CHECK(W_packed.scalar_type() == torch::kUInt8, "W_packed must have dtype torch.uint8");
    TORCH_CHECK(b.scalar_type() == torch::kFloat16, "b must have dtype torch.float16");
    TORCH_CHECK(SZ.scalar_type() == torch::kFloat16, "SZ must have dtype torch.float16");
    TORCH_CHECK(activations.scalar_type() == torch::kFloat16, "activations must have dtype torch.float16");

    TORCH_CHECK(W_packed.device() == b.device(), "W_packed and b must be on the same CUDA device");
    TORCH_CHECK(W_packed.device() == SZ.device(), "W_packed and SZ must be on the same CUDA device");
    TORCH_CHECK(W_packed.device() == activations.device(), "W_packed and activations must be on the same CUDA device");

    TORCH_CHECK(W_packed.dim() == 2, "W_packed must be a 2D tensor of shape [OF_packed, IF]");
    TORCH_CHECK(activations.dim() == 2, "activations must be a 2D tensor of shape [IF, B]");
    TORCH_CHECK(b.dim() == 1 || (b.dim() == 2 && b.size(1) == 1),
                "b must be a 1D tensor of shape [OF] or a 2D tensor of shape [OF, 1]");
    TORCH_CHECK(SZ.dim() == 2, "SZ must be a 2D tensor of shape [IF / group_size, 2 * OF]");

    TORCH_CHECK(group_size > 0, "group_size must be > 0");

    int64_t OF_packed = W_packed.size(0);
    int64_t IF = W_packed.size(1);
    int64_t B = activations.size(1);
    int64_t OF = OF_packed * 2;

    TORCH_CHECK(OF_packed > 0, "W_packed.size(0) must be > 0");
    TORCH_CHECK(IF > 0, "W_packed.size(1) must be > 0");

    TORCH_CHECK(B == 1, "This optimized kernel strictly requires Batch Size B=1");
    TORCH_CHECK(IF % 4 == 0, "Input Features must be a multiple of 4 for 32-bit vectorization");
    TORCH_CHECK(IF % group_size == 0, "Input Features must be divisible by group_size");

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(W_packed.device());
    auto OUT = torch::empty({OF, B}, options);

    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);

    // Each ty now handles 4 packed rows
    int64_t OF_packed_quads = (OF_packed + 3) / 4;
    dim3 blocks((OF_packed_quads + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    w4a16_gemv_vectorized_kernel<<<blocks, threads>>>(
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(SZ.data_ptr<at::Half>()),
        reinterpret_cast<half*>(activations.data_ptr<at::Half>()),
        reinterpret_cast<half*>(OUT.data_ptr<at::Half>()),
        static_cast<int>(OF), static_cast<int>(IF), group_size
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return OUT;
}

PYBIND11_MODULE(w4a16_cuda_ext, m) {
    m.def("forward", &w4a16_forward, "W4A16 GEMV forward pass (CUDA)");
}