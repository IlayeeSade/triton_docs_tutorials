#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 8

// ---------------------------------------------------------
// 1. The VECTORIZED Shared Weight Broadcast Kernel
//    Each threadblock-y lane handles 1 packed row = 4 output rows
//    Each thread processes 8 input features at once
//    For group_size = 128, SZ is refreshed only twice:
//    half 0: steps [0..3]
//    half 1: steps [4..7]
//
//    W_packed layout:
//      shape = [OF / 4, IF]
//      dtype = uint16
//
//    Each uint16 packs 4 adjacent output rows for one feature:
//      bits  3:0   -> row 0
//      bits  7:4   -> row 1
//      bits 11:8   -> row 2
//      bits 15:12  -> row 3
// ---------------------------------------------------------
__global__ void w4a16_gemv_vectorized_kernel(
    const uint16_t* __restrict__ W_packed,
    const half* __restrict__ b,
    const half* __restrict__ SZ,
    const half* __restrict__ activations,
    half* __restrict__ OUT,
    int OF, int IF, int group_size)
{
    static_assert(BLOCK_DIM_X % 32 == 0, "BLOCK_DIM_X must be a multiple of warp size");

    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_TX = BLOCK_DIM_X / WARP_SIZE;
    constexpr int GROUP_SHIFT = 7; // log2(128)

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int lane_id = tx & (WARP_SIZE - 1);
    int warp_id_x = tx >> 5;

    int packed_row_idx = blockIdx.x * BLOCK_DIM_Y + ty;
    int row_idx = packed_row_idx * 4;

    const int OF_packed = OF >> 2;
    if (packed_row_idx >= OF_packed) return;

    // 8 uint16 values per load = 128 bits
    const uint4* W_vec = reinterpret_cast<const uint4*>(W_packed);
    int IF_vec = IF >> 3; // IF / 8

    float partial_acc_0 = 0.0f;
    float partial_acc_1 = 0.0f;
    float partial_acc_2 = 0.0f;
    float partial_acc_3 = 0.0f;

    for (int k_base = 0; k_base < IF_vec; k_base += BLOCK_DIM_X) {
        int k_vec = k_base + tx;

        if (k_vec < IF_vec) {
            // Load 8 activations
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

            // Load 8 packed weights = 8 uint16_t
            uint4 w_chunk = W_vec[packed_row_idx * IF_vec + k_vec];
            const uint16_t* w16 = reinterpret_cast<const uint16_t*>(&w_chunk);

            // ---------------------------------------------------------
            // Half 0: steps 0..3
            // ---------------------------------------------------------
            {
                int feature_idx = (k_vec << 3) + 0;
                int group_idx = feature_idx >> GROUP_SHIFT;

                int base_half = group_idx * (2 * OF) + 2 * row_idx;
                const uint4 packed_sz =
                    reinterpret_cast<const uint4*>(SZ)[base_half >> 3];
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
                for (int step = 0; step < 4; step++) {
                    float act_val;
                    if (step == 0)      act_val = act0;
                    else if (step == 1) act_val = act1;
                    else if (step == 2) act_val = act2;
                    else                act_val = act3;

                    uint16_t w_packed_val = w16[step];

                    uint8_t w0 = (w_packed_val >>  0) & 0x0F;
                    uint8_t w1 = (w_packed_val >>  4) & 0x0F;
                    uint8_t w2 = (w_packed_val >>  8) & 0x0F;
                    uint8_t w3 = (w_packed_val >> 12) & 0x0F;

                    partial_acc_0 += (static_cast<float>(w0) - z_val_0) * s_val_0 * act_val;
                    partial_acc_1 += (static_cast<float>(w1) - z_val_1) * s_val_1 * act_val;
                    partial_acc_2 += (static_cast<float>(w2) - z_val_2) * s_val_2 * act_val;
                    partial_acc_3 += (static_cast<float>(w3) - z_val_3) * s_val_3 * act_val;
                }
            }

            // ---------------------------------------------------------
            // Half 1: steps 4..7
            // ---------------------------------------------------------
            {
                int feature_idx = (k_vec << 3) + 4;
                int group_idx = feature_idx >> GROUP_SHIFT;

                int base_half = group_idx * (2 * OF) + 2 * row_idx;
                const uint4 packed_sz =
                    reinterpret_cast<const uint4*>(SZ)[base_half >> 3];
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
                for (int step = 4; step < 8; step++) {
                    float act_val;
                    if (step == 4)      act_val = act4;
                    else if (step == 5) act_val = act5;
                    else if (step == 6) act_val = act6;
                    else                act_val = act7;

                    uint16_t w_packed_val = w16[step];

                    uint8_t w0 = (w_packed_val >>  0) & 0x0F;
                    uint8_t w1 = (w_packed_val >>  4) & 0x0F;
                    uint8_t w2 = (w_packed_val >>  8) & 0x0F;
                    uint8_t w3 = (w_packed_val >> 12) & 0x0F;

                    partial_acc_0 += (static_cast<float>(w0) - z_val_0) * s_val_0 * act_val;
                    partial_acc_1 += (static_cast<float>(w1) - z_val_1) * s_val_1 * act_val;
                    partial_acc_2 += (static_cast<float>(w2) - z_val_2) * s_val_2 * act_val;
                    partial_acc_3 += (static_cast<float>(w3) - z_val_3) * s_val_3 * act_val;
                }
            }
        }
    }

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

    TORCH_CHECK(W_packed.scalar_type() == torch::kUInt16,
                "W_packed must have dtype torch.uint16");
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

    int64_t OF_packed = W_packed.size(0); // OF / 4
    int64_t IF = W_packed.size(1);
    int64_t OF = OF_packed * 4;
    int64_t B = activations.size(1);

    TORCH_CHECK((OF % 4) == 0,
                "This optimized kernel currently requires OF to be divisible by 4");
    TORCH_CHECK(OF_packed > 0, "W_packed.size(0) must be > 0");
    TORCH_CHECK(IF > 0, "Input features must be > 0");
    TORCH_CHECK(B == 1, "This optimized kernel strictly requires Batch Size B=1");
    TORCH_CHECK(IF % 8 == 0,
                "Input features must be a multiple of 8 for 128-bit vectorization");
    TORCH_CHECK(IF % group_size == 0,
                "Input features must be divisible by group_size");

    TORCH_CHECK(activations.size(0) == IF,
                "activations.shape[0] must match W_packed.shape[1]");
    TORCH_CHECK(b.size(0) == OF,
                "b.shape[0] must match 4 * W_packed.shape[0]");
    TORCH_CHECK(SZ.size(0) == (IF / group_size),
                "SZ.shape[0] must be IF / group_size");
    TORCH_CHECK(SZ.size(1) == (2 * OF),
                "SZ.shape[1] must be 2 * OF");

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat16)
                       .device(W_packed.device());

    auto OUT = torch::empty({OF, B}, options);

    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 blocks((OF_packed + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    w4a16_gemv_vectorized_kernel<<<blocks, threads>>>(
        W_packed.data_ptr<uint16_t>(),
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