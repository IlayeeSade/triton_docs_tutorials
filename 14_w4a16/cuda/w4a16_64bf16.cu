#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 4

__global__ void w4a16_gemv_vectorized_kernel(
    const uint16_t* __restrict__ W_packed,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ SZ,
    const __nv_bfloat16* __restrict__ activations,
    __nv_bfloat16* __restrict__ OUT,
    int OF, int IF, int group_size, int group_shift)
{
    static_assert(BLOCK_DIM_X % 32 == 0, "BLOCK_DIM_X must be a multiple of warp size");

    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_TX = BLOCK_DIM_X / WARP_SIZE;

    // FIX 1: Use runtime group_shift instead of hardcoded GROUP_SHIFT = 6
    // PROBLEM: Original kernel hardcoded GROUP_SHIFT = 6 (assumes group_size = 64)
    // This broke for group_size != 64
    // SOLUTION: Compute GROUP_SHIFT = log2(group_size) at host-side and pass as parameter
    int GROUP_SHIFT = group_shift;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int lane_id = tx & (WARP_SIZE - 1);
    int warp_id_x = tx >> 5;

    int packed_row_idx = blockIdx.x * BLOCK_DIM_Y + ty;
    int row_idx = packed_row_idx * 4;

    const int OF_packed = OF >> 2;
    if (packed_row_idx >= OF_packed) return;

    const uint4* W_vec = reinterpret_cast<const uint4*>(W_packed);
    int IF_vec = IF >> 3; // IF / 8

    // FIX 2: Use double-precision accumulation instead of float
    // PROBLEM: Float32 precision loss with large accumulations (IF=4096)
    // Accumulating 4096 float values loses precision after ~1000 additions
    // With OF=8192 rows, max_err can exceed tolerance
    // SOLUTION: Use double (float64) for accumulation, convert to BF16 at output
    // This maintains full precision during summation, only loses precision at final conversion
    double partial_acc_0 = 0.0;
    double partial_acc_1 = 0.0;
    double partial_acc_2 = 0.0;
    double partial_acc_3 = 0.0;

    for (int k_base = 0; k_base < IF_vec; k_base += BLOCK_DIM_X) {
        int k_vec = k_base + tx;

        if (k_vec < IF_vec) {
            int act_idx = k_vec * 8;
            uint4 packed_acts =
                __ldg(reinterpret_cast<const uint4*>(&activations[act_idx]));
            const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&packed_acts);

            float act0 = __bfloat162float(h[0]);
            float act1 = __bfloat162float(h[1]);
            float act2 = __bfloat162float(h[2]);
            float act3 = __bfloat162float(h[3]);
            float act4 = __bfloat162float(h[4]);
            float act5 = __bfloat162float(h[5]);
            float act6 = __bfloat162float(h[6]);
            float act7 = __bfloat162float(h[7]);

            uint4 w_chunk = W_vec[packed_row_idx * IF_vec + k_vec];
            const uint16_t* w16 = reinterpret_cast<const uint16_t*>(&w_chunk);

            {
                int feature_idx = (k_vec << 3) + 0;
                int group_idx = feature_idx >> GROUP_SHIFT;

                int base_half = group_idx * (2 * OF) + 2 * row_idx;
                const uint4 packed_sz =
                    reinterpret_cast<const uint4*>(SZ)[base_half >> 3];
                const __nv_bfloat16* vals = reinterpret_cast<const __nv_bfloat16*>(&packed_sz);

                float s_val_0 = __bfloat162float(vals[0]);
                float z_val_0 = __bfloat162float(vals[1]);
                float s_val_1 = __bfloat162float(vals[2]);
                float z_val_1 = __bfloat162float(vals[3]);
                float s_val_2 = __bfloat162float(vals[4]);
                float z_val_2 = __bfloat162float(vals[5]);
                float s_val_3 = __bfloat162float(vals[6]);
                float z_val_3 = __bfloat162float(vals[7]);

                #pragma unroll
                for (int step = 0; step < 2; step++) {
                    float act_val = (step == 0) ? act0 : act1;
                    uint16_t w_packed_val = w16[step];

                    uint8_t w0 = (w_packed_val >>  0) & 0x0F;
                    uint8_t w1 = (w_packed_val >>  4) & 0x0F;
                    uint8_t w2 = (w_packed_val >>  8) & 0x0F;
                    uint8_t w3 = (w_packed_val >> 12) & 0x0F;

                    partial_acc_0 += (double)(static_cast<float>(w0) - z_val_0) * s_val_0 * act_val;
                    partial_acc_1 += (double)(static_cast<float>(w1) - z_val_1) * s_val_1 * act_val;
                    partial_acc_2 += (double)(static_cast<float>(w2) - z_val_2) * s_val_2 * act_val;
                    partial_acc_3 += (double)(static_cast<float>(w3) - z_val_3) * s_val_3 * act_val;
                }
            }

            {
                int feature_idx = (k_vec << 3) + 2;
                int group_idx = feature_idx >> GROUP_SHIFT;

                int base_half = group_idx * (2 * OF) + 2 * row_idx;
                const uint4 packed_sz =
                    reinterpret_cast<const uint4*>(SZ)[base_half >> 3];
                const __nv_bfloat16* vals = reinterpret_cast<const __nv_bfloat16*>(&packed_sz);

                float s_val_0 = __bfloat162float(vals[0]);
                float z_val_0 = __bfloat162float(vals[1]);
                float s_val_1 = __bfloat162float(vals[2]);
                float z_val_1 = __bfloat162float(vals[3]);
                float s_val_2 = __bfloat162float(vals[4]);
                float z_val_2 = __bfloat162float(vals[5]);
                float s_val_3 = __bfloat162float(vals[6]);
                float z_val_3 = __bfloat162float(vals[7]);

                #pragma unroll
                for (int step = 2; step < 4; step++) {
                    float act_val = (step == 2) ? act2 : act3;
                    uint16_t w_packed_val = w16[step];

                    uint8_t w0 = (w_packed_val >>  0) & 0x0F;
                    uint8_t w1 = (w_packed_val >>  4) & 0x0F;
                    uint8_t w2 = (w_packed_val >>  8) & 0x0F;
                    uint8_t w3 = (w_packed_val >> 12) & 0x0F;

                    partial_acc_0 += (double)(static_cast<float>(w0) - z_val_0) * s_val_0 * act_val;
                    partial_acc_1 += (double)(static_cast<float>(w1) - z_val_1) * s_val_1 * act_val;
                    partial_acc_2 += (double)(static_cast<float>(w2) - z_val_2) * s_val_2 * act_val;
                    partial_acc_3 += (double)(static_cast<float>(w3) - z_val_3) * s_val_3 * act_val;
                }
            }

            {
                int feature_idx = (k_vec << 3) + 4;
                int group_idx = feature_idx >> GROUP_SHIFT;

                int base_half = group_idx * (2 * OF) + 2 * row_idx;
                const uint4 packed_sz =
                    reinterpret_cast<const uint4*>(SZ)[base_half >> 3];
                const __nv_bfloat16* vals = reinterpret_cast<const __nv_bfloat16*>(&packed_sz);

                float s_val_0 = __bfloat162float(vals[0]);
                float z_val_0 = __bfloat162float(vals[1]);
                float s_val_1 = __bfloat162float(vals[2]);
                float z_val_1 = __bfloat162float(vals[3]);
                float s_val_2 = __bfloat162float(vals[4]);
                float z_val_2 = __bfloat162float(vals[5]);
                float s_val_3 = __bfloat162float(vals[6]);
                float z_val_3 = __bfloat162float(vals[7]);

                #pragma unroll
                for (int step = 4; step < 6; step++) {
                    float act_val = (step == 4) ? act4 : act5;
                    uint16_t w_packed_val = w16[step];

                    uint8_t w0 = (w_packed_val >>  0) & 0x0F;
                    uint8_t w1 = (w_packed_val >>  4) & 0x0F;
                    uint8_t w2 = (w_packed_val >>  8) & 0x0F;
                    uint8_t w3 = (w_packed_val >> 12) & 0x0F;

                    partial_acc_0 += (double)(static_cast<float>(w0) - z_val_0) * s_val_0 * act_val;
                    partial_acc_1 += (double)(static_cast<float>(w1) - z_val_1) * s_val_1 * act_val;
                    partial_acc_2 += (double)(static_cast<float>(w2) - z_val_2) * s_val_2 * act_val;
                    partial_acc_3 += (double)(static_cast<float>(w3) - z_val_3) * s_val_3 * act_val;
                }
            }

            {
                int feature_idx = (k_vec << 3) + 6;
                int group_idx = feature_idx >> GROUP_SHIFT;

                int base_half = group_idx * (2 * OF) + 2 * row_idx;
                const uint4 packed_sz =
                    reinterpret_cast<const uint4*>(SZ)[base_half >> 3];
                const __nv_bfloat16* vals = reinterpret_cast<const __nv_bfloat16*>(&packed_sz);

                float s_val_0 = __bfloat162float(vals[0]);
                float z_val_0 = __bfloat162float(vals[1]);
                float s_val_1 = __bfloat162float(vals[2]);
                float z_val_1 = __bfloat162float(vals[3]);
                float s_val_2 = __bfloat162float(vals[4]);
                float z_val_2 = __bfloat162float(vals[5]);
                float s_val_3 = __bfloat162float(vals[6]);
                float z_val_3 = __bfloat162float(vals[7]);

                #pragma unroll
                for (int step = 6; step < 8; step++) {
                    float act_val = (step == 6) ? act6 : act7;
                    uint16_t w_packed_val = w16[step];

                    uint8_t w0 = (w_packed_val >>  0) & 0x0F;
                    uint8_t w1 = (w_packed_val >>  4) & 0x0F;
                    uint8_t w2 = (w_packed_val >>  8) & 0x0F;
                    uint8_t w3 = (w_packed_val >> 12) & 0x0F;

                    partial_acc_0 += (double)(static_cast<float>(w0) - z_val_0) * s_val_0 * act_val;
                    partial_acc_1 += (double)(static_cast<float>(w1) - z_val_1) * s_val_1 * act_val;
                    partial_acc_2 += (double)(static_cast<float>(w2) - z_val_2) * s_val_2 * act_val;
                    partial_acc_3 += (double)(static_cast<float>(w3) - z_val_3) * s_val_3 * act_val;
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

    __shared__ double warp_sums[BLOCK_DIM_Y][WARPS_PER_TX][4];

    if (lane_id == 0) {
        warp_sums[ty][warp_id_x][0] = partial_acc_0;
        warp_sums[ty][warp_id_x][1] = partial_acc_1;
        warp_sums[ty][warp_id_x][2] = partial_acc_2;
        warp_sums[ty][warp_id_x][3] = partial_acc_3;
    }

    __syncthreads();

    if (warp_id_x == 0) {
        double block_acc_0 = (lane_id < WARPS_PER_TX) ? warp_sums[ty][lane_id][0] : 0.0;
        double block_acc_1 = (lane_id < WARPS_PER_TX) ? warp_sums[ty][lane_id][1] : 0.0;
        double block_acc_2 = (lane_id < WARPS_PER_TX) ? warp_sums[ty][lane_id][2] : 0.0;
        double block_acc_3 = (lane_id < WARPS_PER_TX) ? warp_sums[ty][lane_id][3] : 0.0;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_acc_0 += __shfl_down_sync(0xffffffff, block_acc_0, offset);
            block_acc_1 += __shfl_down_sync(0xffffffff, block_acc_1, offset);
            block_acc_2 += __shfl_down_sync(0xffffffff, block_acc_2, offset);
            block_acc_3 += __shfl_down_sync(0xffffffff, block_acc_3, offset);
        }

        if (lane_id == 0) {
            if (row_idx + 3 < OF) {
                __nv_bfloat16 out_vals[4];
                const uint2 bias_pack = __ldg(reinterpret_cast<const uint2*>(&b[row_idx]));
                const __nv_bfloat16* bias_h = reinterpret_cast<const __nv_bfloat16*>(&bias_pack);

                out_vals[0] = __float2bfloat16_rn(block_acc_0 + __bfloat162float(bias_h[0]));
                out_vals[1] = __float2bfloat16_rn(block_acc_1 + __bfloat162float(bias_h[1]));
                out_vals[2] = __float2bfloat16_rn(block_acc_2 + __bfloat162float(bias_h[2]));
                out_vals[3] = __float2bfloat16_rn(block_acc_3 + __bfloat162float(bias_h[3]));

                *reinterpret_cast<uint2*>(&OUT[row_idx]) =
                    *reinterpret_cast<const uint2*>(out_vals);
            }
        }
    }
}

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
    TORCH_CHECK(b.scalar_type() == torch::kBFloat16,
                "b must have dtype torch.bfloat16");
    TORCH_CHECK(SZ.scalar_type() == torch::kBFloat16,
                "SZ must have dtype torch.bfloat16");
    TORCH_CHECK(activations.scalar_type() == torch::kBFloat16,
                "activations must have dtype torch.bfloat16");

    int64_t OF_packed = W_packed.size(0);
    int64_t IF = W_packed.size(1);
    int64_t OF = OF_packed * 4;
    int64_t B = activations.size(1);

    // FIX 3: Validate that IF is divisible by group_size
    // PROBLEM: If IF is not divisible by group_size, some features fall into
    // incomplete groups with no scale/zero-point data. This causes:
    // - Out-of-bounds reads from SZ array
    // - Incorrect dequantization
    // - Silent data corruption
    // SOLUTION: Add explicit validation at kernel launch time
    TORCH_CHECK(IF % group_size == 0,
                "IF (", IF, ") must be divisible by group_size (", group_size, ")");

    auto options = torch::TensorOptions()
                       .dtype(torch::kBFloat16)
                       .device(W_packed.device());

    auto OUT = torch::empty({OF, B}, options);

    // Compute GROUP_SHIFT = log2(group_size)
    int group_shift = 0;
    int temp = group_size;
    while (temp > 1) {
        temp >>= 1;
        group_shift++;
    }

    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 blocks((OF_packed + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    w4a16_gemv_vectorized_kernel<<<blocks, threads>>>(
        W_packed.data_ptr<uint16_t>(),
        reinterpret_cast<__nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(SZ.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(activations.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(OUT.data_ptr<at::BFloat16>()),
        static_cast<int>(OF),
        static_cast<int>(IF),
        group_size,
        group_shift
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA kernel failed: ", cudaGetErrorString(err));

    return OUT;
}

PYBIND11_MODULE(w4a16_cuda_ext, m) {
    m.def("forward", &w4a16_forward, "W4A16 GEMV forward pass (CUDA, BF16)");
}
