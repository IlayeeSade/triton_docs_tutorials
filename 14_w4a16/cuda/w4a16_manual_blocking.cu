#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BLOCK_DIM_X 32
#define PACKED_ROWS_PER_BLOCK 4

__global__ void w4a16_gemv_vectorized_kernel(
    const uint8_t* __restrict__ W_packed,
    const half* __restrict__ b,
    const half* __restrict__ S,
    const half* __restrict__ Z,
    const half* __restrict__ activations,
    half* __restrict__ OUT,
    int OF, int IF, int group_size)
{
    const int tx = threadIdx.x;
    const int packed_base = blockIdx.x * PACKED_ROWS_PER_BLOCK;
    const int OF_packed = OF / 2;

    const int packed_row_0 = packed_base + 0;
    const int packed_row_1 = packed_base + 1;
    const int packed_row_2 = packed_base + 2;
    const int packed_row_3 = packed_base + 3;

    const bool valid_0 = packed_row_0 < OF_packed;
    const bool valid_1 = packed_row_1 < OF_packed;
    const bool valid_2 = packed_row_2 < OF_packed;
    const bool valid_3 = packed_row_3 < OF_packed;

    const int row0_lo = packed_row_0 * 2;
    const int row0_hi = row0_lo + 1;
    const int row1_lo = packed_row_1 * 2;
    const int row1_hi = row1_lo + 1;
    const int row2_lo = packed_row_2 * 2;
    const int row2_hi = row2_lo + 1;
    const int row3_lo = packed_row_3 * 2;
    const int row3_hi = row3_lo + 1;

    const uint32_t* W_vec = reinterpret_cast<const uint32_t*>(W_packed);
    const int IF_vec = IF / 4;
    const int groups_per_row = IF / group_size;

    float acc0 = 0.0f, acc1 = 0.0f;
    float acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f;
    float acc6 = 0.0f, acc7 = 0.0f;

    for (int k_base = 0; k_base < IF_vec; k_base += BLOCK_DIM_X) {
        const int k_vec = k_base + tx;

        if (k_vec < IF_vec) {
            const int act_idx = k_vec * 4;
            const uint64_t packed_acts =
                __ldg(reinterpret_cast<const uint64_t*>(&activations[act_idx]));
            const half* h = reinterpret_cast<const half*>(&packed_acts);

            const float act0 = __half2float(h[0]);
            const float act1 = __half2float(h[1]);
            const float act2 = __half2float(h[2]);
            const float act3 = __half2float(h[3]);

            const int group_idx = (k_vec * 4) / group_size;

            if (valid_0) {
                const uint32_t w_chunk = W_vec[packed_row_0 * IF_vec + k_vec];
                const float s_lo = __half2float(__ldg(&S[row0_lo * groups_per_row + group_idx]));
                const float s_hi = __half2float(__ldg(&S[row0_hi * groups_per_row + group_idx]));
                const float z_lo = __half2float(__ldg(&Z[row0_lo * groups_per_row + group_idx]));
                const float z_hi = __half2float(__ldg(&Z[row0_hi * groups_per_row + group_idx]));

                const uint8_t b0 = (w_chunk >>  0) & 0xFF;
                const uint8_t b1 = (w_chunk >>  8) & 0xFF;
                const uint8_t b2 = (w_chunk >> 16) & 0xFF;
                const uint8_t b3 = (w_chunk >> 24) & 0xFF;

                acc0 += ((float)( b0       & 0x0F) - z_lo) * s_lo * act0;
                acc1 += ((float)((b0 >> 4) & 0x0F) - z_hi) * s_hi * act0;

                acc0 += ((float)( b1       & 0x0F) - z_lo) * s_lo * act1;
                acc1 += ((float)((b1 >> 4) & 0x0F) - z_hi) * s_hi * act1;

                acc0 += ((float)( b2       & 0x0F) - z_lo) * s_lo * act2;
                acc1 += ((float)((b2 >> 4) & 0x0F) - z_hi) * s_hi * act2;

                acc0 += ((float)( b3       & 0x0F) - z_lo) * s_lo * act3;
                acc1 += ((float)((b3 >> 4) & 0x0F) - z_hi) * s_hi * act3;
            }

            if (valid_1) {
                const uint32_t w_chunk = W_vec[packed_row_1 * IF_vec + k_vec];
                const float s_lo = __half2float(__ldg(&S[row1_lo * groups_per_row + group_idx]));
                const float s_hi = __half2float(__ldg(&S[row1_hi * groups_per_row + group_idx]));
                const float z_lo = __half2float(__ldg(&Z[row1_lo * groups_per_row + group_idx]));
                const float z_hi = __half2float(__ldg(&Z[row1_hi * groups_per_row + group_idx]));

                const uint8_t b0 = (w_chunk >>  0) & 0xFF;
                const uint8_t b1 = (w_chunk >>  8) & 0xFF;
                const uint8_t b2 = (w_chunk >> 16) & 0xFF;
                const uint8_t b3 = (w_chunk >> 24) & 0xFF;

                acc2 += ((float)( b0       & 0x0F) - z_lo) * s_lo * act0;
                acc3 += ((float)((b0 >> 4) & 0x0F) - z_hi) * s_hi * act0;

                acc2 += ((float)( b1       & 0x0F) - z_lo) * s_lo * act1;
                acc3 += ((float)((b1 >> 4) & 0x0F) - z_hi) * s_hi * act1;

                acc2 += ((float)( b2       & 0x0F) - z_lo) * s_lo * act2;
                acc3 += ((float)((b2 >> 4) & 0x0F) - z_hi) * s_hi * act2;

                acc2 += ((float)( b3       & 0x0F) - z_lo) * s_lo * act3;
                acc3 += ((float)((b3 >> 4) & 0x0F) - z_hi) * s_hi * act3;
            }

            if (valid_2) {
                const uint32_t w_chunk = W_vec[packed_row_2 * IF_vec + k_vec];
                const float s_lo = __half2float(__ldg(&S[row2_lo * groups_per_row + group_idx]));
                const float s_hi = __half2float(__ldg(&S[row2_hi * groups_per_row + group_idx]));
                const float z_lo = __half2float(__ldg(&Z[row2_lo * groups_per_row + group_idx]));
                const float z_hi = __half2float(__ldg(&Z[row2_hi * groups_per_row + group_idx]));

                const uint8_t b0 = (w_chunk >>  0) & 0xFF;
                const uint8_t b1 = (w_chunk >>  8) & 0xFF;
                const uint8_t b2 = (w_chunk >> 16) & 0xFF;
                const uint8_t b3 = (w_chunk >> 24) & 0xFF;

                acc4 += ((float)( b0       & 0x0F) - z_lo) * s_lo * act0;
                acc5 += ((float)((b0 >> 4) & 0x0F) - z_hi) * s_hi * act0;

                acc4 += ((float)( b1       & 0x0F) - z_lo) * s_lo * act1;
                acc5 += ((float)((b1 >> 4) & 0x0F) - z_hi) * s_hi * act1;

                acc4 += ((float)( b2       & 0x0F) - z_lo) * s_lo * act2;
                acc5 += ((float)((b2 >> 4) & 0x0F) - z_hi) * s_hi * act2;

                acc4 += ((float)( b3       & 0x0F) - z_lo) * s_lo * act3;
                acc5 += ((float)((b3 >> 4) & 0x0F) - z_hi) * s_hi * act3;
            }

            if (valid_3) {
                const uint32_t w_chunk = W_vec[packed_row_3 * IF_vec + k_vec];
                const float s_lo = __half2float(__ldg(&S[row3_lo * groups_per_row + group_idx]));
                const float s_hi = __half2float(__ldg(&S[row3_hi * groups_per_row + group_idx]));
                const float z_lo = __half2float(__ldg(&Z[row3_lo * groups_per_row + group_idx]));
                const float z_hi = __half2float(__ldg(&Z[row3_hi * groups_per_row + group_idx]));

                const uint8_t b0 = (w_chunk >>  0) & 0xFF;
                const uint8_t b1 = (w_chunk >>  8) & 0xFF;
                const uint8_t b2 = (w_chunk >> 16) & 0xFF;
                const uint8_t b3 = (w_chunk >> 24) & 0xFF;

                acc6 += ((float)( b0       & 0x0F) - z_lo) * s_lo * act0;
                acc7 += ((float)((b0 >> 4) & 0x0F) - z_hi) * s_hi * act0;

                acc6 += ((float)( b1       & 0x0F) - z_lo) * s_lo * act1;
                acc7 += ((float)((b1 >> 4) & 0x0F) - z_hi) * s_hi * act1;

                acc6 += ((float)( b2       & 0x0F) - z_lo) * s_lo * act2;
                acc7 += ((float)((b2 >> 4) & 0x0F) - z_hi) * s_hi * act2;

                acc6 += ((float)( b3       & 0x0F) - z_lo) * s_lo * act3;
                acc7 += ((float)((b3 >> 4) & 0x0F) - z_hi) * s_hi * act3;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc0 += __shfl_down_sync(0xffffffff, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffff, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffff, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffff, acc3, offset);
        acc4 += __shfl_down_sync(0xffffffff, acc4, offset);
        acc5 += __shfl_down_sync(0xffffffff, acc5, offset);
        acc6 += __shfl_down_sync(0xffffffff, acc6, offset);
        acc7 += __shfl_down_sync(0xffffffff, acc7, offset);
    }

    if (tx == 0) {
        if (valid_0) {
            half2 bias_h2 = __ldg(reinterpret_cast<const half2*>(&b[row0_lo]));
            float2 bias_f2 = __half22float2(bias_h2);
            *reinterpret_cast<half2*>(&OUT[row0_lo]) =
                __floats2half2_rn(acc0 + bias_f2.x, acc1 + bias_f2.y);
        }

        if (valid_1) {
            half2 bias_h2 = __ldg(reinterpret_cast<const half2*>(&b[row1_lo]));
            float2 bias_f2 = __half22float2(bias_h2);
            *reinterpret_cast<half2*>(&OUT[row1_lo]) =
                __floats2half2_rn(acc2 + bias_f2.x, acc3 + bias_f2.y);
        }

        if (valid_2) {
            half2 bias_h2 = __ldg(reinterpret_cast<const half2*>(&b[row2_lo]));
            float2 bias_f2 = __half22float2(bias_h2);
            *reinterpret_cast<half2*>(&OUT[row2_lo]) =
                __floats2half2_rn(acc4 + bias_f2.x, acc5 + bias_f2.y);
        }

        if (valid_3) {
            half2 bias_h2 = __ldg(reinterpret_cast<const half2*>(&b[row3_lo]));
            float2 bias_f2 = __half22float2(bias_h2);
            *reinterpret_cast<half2*>(&OUT[row3_lo]) =
                __floats2half2_rn(acc6 + bias_f2.x, acc7 + bias_f2.y);
        }
    }
}

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

    const int OF_packed = W_packed.size(0);
    const int IF = W_packed.size(1);
    const int B = activations.size(1);
    const int OF = OF_packed * 2;

    TORCH_CHECK(B == 1, "This optimized kernel strictly requires Batch Size B=1");
    TORCH_CHECK(IF % 4 == 0, "Input Features must be a multiple of 4 for 32-bit vectorization");

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(W_packed.device());
    auto OUT = torch::empty({OF, B}, options);

    dim3 threads(BLOCK_DIM_X);
    dim3 blocks((OF_packed + PACKED_ROWS_PER_BLOCK - 1) / PACKED_ROWS_PER_BLOCK);

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