#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 4

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

    // Each ty processes 2 packed rows = 4 output rows
    int packed_pair_idx = blockIdx.x * BLOCK_DIM_Y + ty;
    int current_of_packed_0 = packed_pair_idx * 2;
    int current_of_packed_1 = current_of_packed_0 + 1;

    int row_idx = current_of_packed_0 * 2;

    const int OF_packed = OF / 2;
    if (current_of_packed_0 >= OF_packed) return;

    // We now process 8 input features per thread per iteration
    const int IF_vec8 = IF / 8;

    float partial_acc_0 = 0.0f;
    float partial_acc_1 = 0.0f;
    float partial_acc_2 = 0.0f;
    float partial_acc_3 = 0.0f;

    const bool has_second_packed_row = (current_of_packed_1 < OF_packed);

    for (int k_base = 0; k_base < IF_vec8; k_base += BLOCK_DIM_X) {
        int k_vec8 = k_base + tx;

        if (k_vec8 < IF_vec8) {
            // -------------------------------------------------
            // 1) Load 8 activations = 16 bytes = uint4
            // -------------------------------------------------
            int act_idx = k_vec8 * 8;

            uint4 packed_acts = __ldg(reinterpret_cast<const uint4*>(&activations[act_idx]));
            const half* h = reinterpret_cast<const half*>(&packed_acts);

            float act0 = __half2float(h[0]);
            float act1 = __half2float(h[1]);
            float act2 = __half2float(h[2]);
            float act3 = __half2float(h[3]);
            float act4 = __half2float(h[4]);
            float act5 = __half2float(h[5]);
            float act6 = __half2float(h[6]);
            float act7 = __half2float(h[7]);

            // -------------------------------------------------
            // 2) Load 8 packed weights per packed row = 8 bytes
            //    Each byte contains 2x 4-bit weights
            // -------------------------------------------------
            const uint64_t* W_u64 = reinterpret_cast<const uint64_t*>(W_packed);

            // Row stride in uint64_t units: IF bytes / 8
            uint64_t w_chunk_0 = W_u64[current_of_packed_0 * IF_vec8 + k_vec8];
            uint64_t w_chunk_1 = 0;
            if (has_second_packed_row) {
                w_chunk_1 = W_u64[current_of_packed_1 * IF_vec8 + k_vec8];
            }

            // Since group_size == 256 and each thread handles 8 features,
            // one warp covers exactly one full group: 32 * 8 = 256
            int group_idx = (k_vec8 * 8) / group_size;

            // Load [s0,z0,s1,z1,s2,z2,s3,z3] for rows row_idx..row_idx+3
            int base_half = group_idx * (2 * OF) + 2 * row_idx;

            const uint4 packed_sz = reinterpret_cast<const uint4*>(SZ)[base_half >> 3];
            const half* vals = reinterpret_cast<const half*>(&packed_sz);

            float s_val_0 = __half2float(vals[0]);
            float z_val_0 = __half2float(vals[1]);
            float s_val_1 = __half2float(vals[2]);
            float z_val_1 = __half2float(vals[3]);
            float s_val_2 = __half2float(vals[4]);
            float z_val_2 = __half2float(vals[5]);
            float s_val_3 = __half2float(vals[6]);
            float z_val_3 = __half2float(vals[7]);

            #pragma unroll
            for (int step = 0; step < 8; step++) {
                float act_val =
                    (step == 0) ? act0 :
                    (step == 1) ? act1 :
                    (step == 2) ? act2 :
                    (step == 3) ? act3 :
                    (step == 4) ? act4 :
                    (step == 5) ? act5 :
                    (step == 6) ? act6 : act7;

                // First packed row -> rows row_idx, row_idx+1
                uint8_t w_packed_val_0 = (w_chunk_0 >> (step * 8)) & 0xFF;
                uint8_t w0_lo =  w_packed_val_0       & 0x0F;
                uint8_t w0_hi = (w_packed_val_0 >> 4) & 0x0F;

                float w_deq_0 = (static_cast<float>(w0_lo) - z_val_0) * s_val_0;
                float w_deq_1 = (static_cast<float>(w0_hi) - z_val_1) * s_val_1;

                partial_acc_0 += w_deq_0 * act_val;
                partial_acc_1 += w_deq_1 * act_val;

                // Second packed row -> rows row_idx+2, row_idx+3
                if (has_second_packed_row) {
                    uint8_t w_packed_val_1 = (w_chunk_1 >> (step * 8)) & 0xFF;
                    uint8_t w1_lo =  w_packed_val_1       & 0x0F;
                    uint8_t w1_hi = (w_packed_val_1 >> 4) & 0x0F;

                    float w_deq_2 = (static_cast<float>(w1_lo) - z_val_2) * s_val_2;
                    float w_deq_3 = (static_cast<float>(w1_hi) - z_val_3) * s_val_3;

                    partial_acc_2 += w_deq_2 * act_val;
                    partial_acc_3 += w_deq_3 * act_val;
                }
            }
        }
    }

    // ---------------------------------------------------------
    // Reduction
    // ---------------------------------------------------------
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_acc_0 += __shfl_down_sync(0xffffffff, partial_acc_0, offset);
        partial_acc_1 += __shfl_down_sync(0xffffffff, partial_acc_1, offset);
        partial_acc_2 += __shfl_down_sync(0xffffffff, partial_acc_2, offset);
        partial_acc_3 += __shfl_down_sync(0xffffffff, partial_acc_3, offset);
    }

    constexpr int NUM_WARPS = BLOCK_DIM_X / 32;
    __shared__ float shm_reduce[BLOCK_DIM_Y][NUM_WARPS][4];

    int warp_id = tx / 32;
    int lane_id = tx % 32;

    if (lane_id == 0) {
        shm_reduce[ty][warp_id][0] = partial_acc_0;
        shm_reduce[ty][warp_id][1] = partial_acc_1;
        shm_reduce[ty][warp_id][2] = partial_acc_2;
        shm_reduce[ty][warp_id][3] = partial_acc_3;
    }
    __syncthreads();

    if (tx == 0) {
        partial_acc_0 = 0.0f;
        partial_acc_1 = 0.0f;
        partial_acc_2 = 0.0f;
        partial_acc_3 = 0.0f;

        #pragma unroll
        for (int i = 0; i < NUM_WARPS; i++) {
            partial_acc_0 += shm_reduce[ty][i][0];
            partial_acc_1 += shm_reduce[ty][i][1];
            partial_acc_2 += shm_reduce[ty][i][2];
            partial_acc_3 += shm_reduce[ty][i][3];
        }

        // Store first 2 rows
        half2 bias_h2_0 = __ldg(reinterpret_cast<const half2*>(&b[row_idx]));
        float2 bias_f2_0 = __half22float2(bias_h2_0);

        half2 out_h2_0 = __floats2half2_rn(
            partial_acc_0 + bias_f2_0.x,
            partial_acc_1 + bias_f2_0.y
        );
        *reinterpret_cast<half2*>(&OUT[row_idx]) = out_h2_0;

        // Store next 2 rows if valid
        if (row_idx + 2 < OF) {
            half2 bias_h2_1 = __ldg(reinterpret_cast<const half2*>(&b[row_idx + 2]));
            float2 bias_f2_1 = __half22float2(bias_h2_1);

            half2 out_h2_1 = __floats2half2_rn(
                partial_acc_2 + bias_f2_1.x,
                partial_acc_3 + bias_f2_1.y
            );
            *reinterpret_cast<half2*>(&OUT[row_idx + 2]) = out_h2_1;
        }
    }
}

// ---------------------------------------------------------
// PyTorch wrapper
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

    TORCH_CHECK(group_size == 256, "This version assumes group_size == 256");

    int64_t OF_packed = W_packed.size(0);
    int64_t IF = W_packed.size(1);
    int64_t B = activations.size(1);
    int64_t OF = OF_packed * 2;

    TORCH_CHECK(OF_packed > 0, "W_packed.size(0) must be > 0");
    TORCH_CHECK(IF > 0, "W_packed.size(1) must be > 0");

    TORCH_CHECK(B == 1, "This optimized kernel strictly requires Batch Size B=1");
    TORCH_CHECK(IF % 8 == 0, "Input Features must be a multiple of 8 for 64-bit weight / 128-bit activation vectorization");
    TORCH_CHECK(IF % group_size == 0, "Input Features must be divisible by group_size");

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(W_packed.device());
    auto OUT = torch::empty({OF, B}, options);

    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);

    int64_t OF_packed_pairs = (OF_packed + 1) / 2;
    dim3 blocks((OF_packed_pairs + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

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