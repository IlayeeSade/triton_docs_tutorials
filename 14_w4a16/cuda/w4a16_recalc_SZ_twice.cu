#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 8

// ---------------------------------------------------------
// 1. The VECTORIZED Shared Weight Broadcast Kernel
//    Each threadblock-y lane still handles 2 packed rows = 4 output rows
//    Each thread now processes 8 input features at once
//    BLOCK_DIM_X can now be multiple warps (here: 64 = 2 warps)
// ---------------------------------------------------------
__global__ void w4a16_gemv_vectorized_kernel(
    const uint8_t* __restrict__ W_packed,
    const half* __restrict__ b,
    const half* __restrict__ SZ,
    const half* __restrict__ activations,
    half* __restrict__ OUT,
    int OF, int IF, int group_size)
{
    static_assert(BLOCK_DIM_X % 32 == 0, "BLOCK_DIM_X must be a multiple of warp size");

    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_TX = BLOCK_DIM_X / WARP_SIZE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int lane_id = tx & (WARP_SIZE - 1);
    int warp_id_x = tx >> 5; // which warp along x

    // Still 2 packed rows = 4 real rows
    int packed_pair_idx = blockIdx.x * BLOCK_DIM_Y + ty;
    int current_of_packed_0 = packed_pair_idx * 2;
    int current_of_packed_1 = current_of_packed_0 + 1;
    int row_idx = current_of_packed_0 * 2; // first of 4 output rows

    const int OF_packed = OF >> 1;
    if (current_of_packed_0 >= OF_packed) return;

    // Now each chunk is 64 bits = 8 bytes = 8 input features per packed row
    const uint64_t* W_vec = reinterpret_cast<const uint64_t*>(W_packed);
    int IF_vec = IF >> 3; // IF / 8

    float partial_acc_0 = 0.0f; // row_idx + 0
    float partial_acc_1 = 0.0f; // row_idx + 1
    float partial_acc_2 = 0.0f; // row_idx + 2
    float partial_acc_3 = 0.0f; // row_idx + 3

    for (int k_base = 0; k_base < IF_vec; k_base += BLOCK_DIM_X) {
        int k_vec = k_base + tx;

        if (k_vec < IF_vec) {
            // Load 8 activations = 128 bits
            int act_idx = k_vec * 8;
            uint4 packed_acts =
                __ldg(reinterpret_cast<const uint4*>(&activations[act_idx]));
            const half* h = reinterpret_cast<const half*>(&packed_acts);

            float act0 = __half2float(h[0]);
            float act1 = __half2float(h[1]);
            float act2 = __half2float(h[2]);
            float act3 = __half2float(h[3]);
            float act4 = __half2float(h[4]);
            float act5 = __half2float(h[5]);
            float act6 = __half2float(h[6]);
            float act7 = __half2float(h[7]);

            // Load 2 packed rows, each load is 64-bit
            uint64_t w_chunk_0 = W_vec[current_of_packed_0 * IF_vec + k_vec];
            uint64_t w_chunk_1 = W_vec[current_of_packed_1 * IF_vec + k_vec];

            constexpr int GROUP_SHIFT = 7; // log2(128)

            // ---------------------------------------------------------
            // First half: steps 0..3
            // These correspond to features [k_vec*8 + 0 .. k_vec*8 + 3]
            // ---------------------------------------------------------
            {
                int feature_idx_0 = (k_vec << 3); // k_vec * 8
                int group_idx_0 = feature_idx_0 >> GROUP_SHIFT;

                // Load [s0,z0,s1,z1,s2,z2,s3,z3] for 4 rows
                int base_half_0 = group_idx_0 * (2 * OF) + 2 * row_idx;
                const uint4 packed_sz_0 =
                    reinterpret_cast<const uint4*>(SZ)[base_half_0 >> 3];
                const half* vals_0 = reinterpret_cast<const half*>(&packed_sz_0);

                float s_val_0_0 = __half2float(vals_0[0]);
                float z_val_0_0 = __half2float(vals_0[1]);
                float s_val_1_0 = __half2float(vals_0[2]);
                float z_val_1_0 = __half2float(vals_0[3]);
                float s_val_2_0 = __half2float(vals_0[4]);
                float z_val_2_0 = __half2float(vals_0[5]);
                float s_val_3_0 = __half2float(vals_0[6]);
                float z_val_3_0 = __half2float(vals_0[7]);

                #pragma unroll
                for (int step = 0; step < 4; step++) {
                    float act_val =
                        (step == 0 ? act0 :
                         step == 1 ? act1 :
                         step == 2 ? act2 : act3);

                    // First packed row -> rows row_idx, row_idx+1
                    uint8_t w_packed_val_0 = (w_chunk_0 >> (step * 8)) & 0xFF;
                    uint8_t w0_lo = w_packed_val_0 & 0x0F;
                    uint8_t w0_hi = (w_packed_val_0 >> 4) & 0x0F;

                    float w_deq_0 = (static_cast<float>(w0_lo) - z_val_0_0) * s_val_0_0;
                    float w_deq_1 = (static_cast<float>(w0_hi) - z_val_1_0) * s_val_1_0;

                    partial_acc_0 += w_deq_0 * act_val;
                    partial_acc_1 += w_deq_1 * act_val;

                    // Second packed row -> rows row_idx+2, row_idx+3
                    uint8_t w_packed_val_1 = (w_chunk_1 >> (step * 8)) & 0xFF;
                    uint8_t w1_lo = w_packed_val_1 & 0x0F;
                    uint8_t w1_hi = (w_packed_val_1 >> 4) & 0x0F;

                    float w_deq_2 = (static_cast<float>(w1_lo) - z_val_2_0) * s_val_2_0;
                    float w_deq_3 = (static_cast<float>(w1_hi) - z_val_3_0) * s_val_3_0;

                    partial_acc_2 += w_deq_2 * act_val;
                    partial_acc_3 += w_deq_3 * act_val;
                }
            }

            // ---------------------------------------------------------
            // Second half: steps 4..7
            // These correspond to features [k_vec*8 + 4 .. k_vec*8 + 7]
            // For group_size = 128, SZ may change here, so reload once.
            // ---------------------------------------------------------
            {
                int feature_idx_1 = (k_vec << 3) + 4; // k_vec * 8 + 4
                int group_idx_1 = feature_idx_1 >> GROUP_SHIFT;

                // Load [s0,z0,s1,z1,s2,z2,s3,z3] for 4 rows
                int base_half_1 = group_idx_1 * (2 * OF) + 2 * row_idx;
                const uint4 packed_sz_1 =
                    reinterpret_cast<const uint4*>(SZ)[base_half_1 >> 3];
                const half* vals_1 = reinterpret_cast<const half*>(&packed_sz_1);

                float s_val_0_1 = __half2float(vals_1[0]);
                float z_val_0_1 = __half2float(vals_1[1]);
                float s_val_1_1 = __half2float(vals_1[2]);
                float z_val_1_1 = __half2float(vals_1[3]);
                float s_val_2_1 = __half2float(vals_1[4]);
                float z_val_2_1 = __half2float(vals_1[5]);
                float s_val_3_1 = __half2float(vals_1[6]);
                float z_val_3_1 = __half2float(vals_1[7]);

                #pragma unroll
                for (int step = 4; step < 8; step++) {
                    float act_val =
                        (step == 4 ? act4 :
                         step == 5 ? act5 :
                         step == 6 ? act6 : act7);

                    // First packed row -> rows row_idx, row_idx+1
                    uint8_t w_packed_val_0 = (w_chunk_0 >> (step * 8)) & 0xFF;
                    uint8_t w0_lo = w_packed_val_0 & 0x0F;
                    uint8_t w0_hi = (w_packed_val_0 >> 4) & 0x0F;

                    float w_deq_0 = (static_cast<float>(w0_lo) - z_val_0_1) * s_val_0_1;
                    float w_deq_1 = (static_cast<float>(w0_hi) - z_val_1_1) * s_val_1_1;

                    partial_acc_0 += w_deq_0 * act_val;
                    partial_acc_1 += w_deq_1 * act_val;

                    // Second packed row -> rows row_idx+2, row_idx+3
                    uint8_t w_packed_val_1 = (w_chunk_1 >> (step * 8)) & 0xFF;
                    uint8_t w1_lo = w_packed_val_1 & 0x0F;
                    uint8_t w1_hi = (w_packed_val_1 >> 4) & 0x0F;

                    float w_deq_2 = (static_cast<float>(w1_lo) - z_val_2_1) * s_val_2_1;
                    float w_deq_3 = (static_cast<float>(w1_hi) - z_val_3_1) * s_val_3_1;

                    partial_acc_2 += w_deq_2 * act_val;
                    partial_acc_3 += w_deq_3 * act_val;
                }
            }
        }
    }

    // ---------------------------------------------------------
    // REDUCTION STAGE
    // 1. Reduce inside each warp
    // 2. Store one partial sum per warp in shared memory
    // 3. Let the first warp reduce those warp sums
    // ---------------------------------------------------------
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial_acc_0 += __shfl_down_sync(0xffffffff, partial_acc_0, offset);
        partial_acc_1 += __shfl_down_sync(0xffffffff, partial_acc_1, offset);
        partial_acc_2 += __shfl_down_sync(0xffffffff, partial_acc_2, offset);
        partial_acc_3 += __shfl_down_sync(0xffffffff, partial_acc_3, offset);
    }

    __shared__ float warp_sums[BLOCK_DIM_Y][WARPS_PER_TX][4];

    if (lane_id == 0) {
        warp_sums[ty][warp_id_x][0] = partial_acc_0;
        warp_sums[ty][warp_id_x][1] = partial_acc_1;
        warp_sums[ty][warp_id_x][2] = partial_acc_2;
        warp_sums[ty][warp_id_x][3] = partial_acc_3;
    }

    __syncthreads();

    if (warp_id_x == 0) {
        float block_acc_0 = (lane_id < WARPS_PER_TX) ? warp_sums[ty][lane_id][0] : 0.0f;
        float block_acc_1 = (lane_id < WARPS_PER_TX) ? warp_sums[ty][lane_id][1] : 0.0f;
        float block_acc_2 = (lane_id < WARPS_PER_TX) ? warp_sums[ty][lane_id][2] : 0.0f;
        float block_acc_3 = (lane_id < WARPS_PER_TX) ? warp_sums[ty][lane_id][3] : 0.0f;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_acc_0 += __shfl_down_sync(0xffffffff, block_acc_0, offset);
            block_acc_1 += __shfl_down_sync(0xffffffff, block_acc_1, offset);
            block_acc_2 += __shfl_down_sync(0xffffffff, block_acc_2, offset);
            block_acc_3 += __shfl_down_sync(0xffffffff, block_acc_3, offset);
        }

        if (lane_id == 0) {
            if (row_idx + 3 < OF) {
                const uint2 bias_pack = __ldg(reinterpret_cast<const uint2*>(&b[row_idx]));
                const half* bias_h = reinterpret_cast<const half*>(&bias_pack);

                half out_vals[4];
                out_vals[0] = __float2half_rn(block_acc_0 + __half2float(bias_h[0]));
                out_vals[1] = __float2half_rn(block_acc_1 + __half2float(bias_h[1]));
                out_vals[2] = __float2half_rn(block_acc_2 + __half2float(bias_h[2]));
                out_vals[3] = __float2half_rn(block_acc_3 + __half2float(bias_h[3]));

                *reinterpret_cast<uint2*>(&OUT[row_idx]) =
                    *reinterpret_cast<const uint2*>(out_vals);
            }
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

    TORCH_CHECK(W_packed.scalar_type() == torch::kUInt8,
                "W_packed must have dtype torch.uint8");
    TORCH_CHECK(b.scalar_type() == torch::kFloat16,
                "b must have dtype torch.float16");
    TORCH_CHECK(SZ.scalar_type() == torch::kFloat16,
                "SZ must have dtype torch.float16");
    TORCH_CHECK(activations.scalar_type() == torch::kFloat16,
                "activations must have dtype torch.float16");

    TORCH_CHECK(W_packed.device() == b.device(),
                "W_packed and b must be on the same CUDA device");
    TORCH_CHECK(W_packed.device() == SZ.device(),
                "W_packed and SZ must be on the same CUDA device");
    TORCH_CHECK(W_packed.device() == activations.device(),
                "W_packed and activations must be on the same CUDA device");

    TORCH_CHECK(W_packed.dim() == 2,
                "W_packed must be a 2D tensor");
    TORCH_CHECK(activations.dim() == 2,
                "activations must be a 2D tensor of shape [IF, B]");
    TORCH_CHECK(
        b.dim() == 1 || (b.dim() == 2 && b.size(1) == 1),
        "b must be a 1D tensor of shape [OF] or a 2D tensor of shape [OF, 1]");
    TORCH_CHECK(SZ.dim() == 2,
                "SZ must be a 2D tensor of shape [IF / group_size, 2 * OF]");

    TORCH_CHECK(group_size > 0, "group_size must be > 0");
    TORCH_CHECK(group_size == 128,
                "This optimized kernel currently requires group_size == 128");

    int64_t OF_packed = W_packed.size(0);
    int64_t IF = W_packed.size(1);
    int64_t OF = OF_packed * 2;
    int64_t B = activations.size(1);
    
    TORCH_CHECK((OF % 4) == 0,
                "This optimized kernel currently requires OF to be divisible by 4");
    TORCH_CHECK(OF_packed > 0, "W_packed.size(0) must be > 0");
    TORCH_CHECK(IF > 0, "Input features must be > 0");
    TORCH_CHECK(B == 1, "This optimized kernel strictly requires Batch Size B=1");
    TORCH_CHECK(IF % 8 == 0,
                "Input Features must be a multiple of 8 for 64-bit vectorization");
    TORCH_CHECK(IF % group_size == 0,
                "Input Features must be divisible by group_size");

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat16)
                       .device(W_packed.device());

    auto OUT = torch::empty({OF, B}, options);

    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);

    // Each ty still handles 2 packed rows
    int64_t OF_packed_pairs = (OF_packed + 1) / 2;
    dim3 blocks((OF_packed_pairs + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    w4a16_gemv_vectorized_kernel<<<blocks, threads>>>(
        W_packed.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(SZ.data_ptr<at::Half>()),
        reinterpret_cast<half*>(activations.data_ptr<at::Half>()),
        reinterpret_cast<half*>(OUT.data_ptr<at::Half>()),
        static_cast<int>(OF),
        static_cast<int>(IF),
        group_size
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA kernel failed: ", cudaGetErrorString(err));

    return OUT;
}

PYBIND11_MODULE(w4a16_cuda_ext, m) {
    m.def("forward", &w4a16_forward, "W4A16 GEMV forward pass (CUDA)");
}