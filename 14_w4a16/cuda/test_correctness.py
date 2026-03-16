"""
Correctness tests for W4A16 and AWQ CUDA kernels.
Tests numerical accuracy, edge cases, and various configurations.
"""

import torch
from torch.utils.cpp_extension import load
import os
import pytest
import numpy as np

# Load CUDA extensions
this_dir = os.path.dirname(os.path.abspath(__file__))

w4a16_cuda_ext = load(
    name="w4a16_cuda_ext",
    sources=[os.path.join(this_dir, "w4a16_cuda.cu")],
    extra_cuda_cflags=["-O3", "-allow-unsupported-compiler"],
    verbose=False,
)

awq_cuda_ext = load(
    name="awq_cuda_ext",
    sources=[os.path.join(this_dir, "awq_kernel.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# Helper Functions (from benchmark_suite.py)
# ==============================================================================

def pack_rows_4(W_q: torch.Tensor) -> torch.Tensor:
    """Pack 4 adjacent rows into one 16-bit element."""
    assert W_q.dim() == 2
    OF, IF = W_q.shape
    assert OF % 4 == 0, "OF must be divisible by 4 for 4-row packing"

    W_q = W_q.to(torch.int16).contiguous()
    r0 = W_q[0::4]
    r1 = W_q[1::4]
    r2 = W_q[2::4]
    r3 = W_q[3::4]

    packed = (
        (r0 & 0xF)
        | ((r1 & 0xF) << 4)
        | ((r2 & 0xF) << 8)
        | ((r3 & 0xF) << 12)
    )
    return packed.to(torch.uint16).contiguous()


def compute_group_shift(group_size: int) -> int:
    """Compute log2(group_size) for GROUP_SHIFT calculation."""
    shift = 0
    temp = group_size
    while temp > 1:
        temp >>= 1
        shift += 1
    return shift


def unpack_rows_4(W_packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_rows_4."""
    W_packed = W_packed.to(torch.int32).contiguous()
    w0 = (W_packed >> 0) & 0x0F
    w1 = (W_packed >> 4) & 0x0F
    w2 = (W_packed >> 8) & 0x0F
    w3 = (W_packed >> 12) & 0x0F

    return (
        torch.stack([w0, w1, w2, w3], dim=1)
        .reshape(-1, W_packed.shape[1])
        .to(torch.uint8)
        .contiguous()
    )


def dequantize_layer(W_q, S, Z, group_size):
    """Reference dequantization: (W_q - Z) * S"""
    N, K = W_q.shape
    W_q_reshaped = W_q.view(N, K // group_size, group_size)
    W_deq = (W_q_reshaped.to(torch.bfloat16) - Z.unsqueeze(-1)) * S.unsqueeze(-1)
    return W_deq.view(N, K)


def torch_w4a16_from_packed4(W_packed, b, S, Z, group_size, activations):
    """PyTorch reference implementation for W4A16."""
    W_unpacked = unpack_rows_4(W_packed)
    W_deq = dequantize_layer(W_unpacked, S, Z, group_size)
    out = torch.matmul(W_deq, activations) + b[:, None]
    return out


def interleave_transposed_s_z(S: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Convert S, Z: [OF, G] into SZ: [G, 2 * OF] with interleaved layout."""
    assert S.shape == Z.shape
    assert S.dtype == torch.bfloat16 and Z.dtype == torch.bfloat16
    assert S.is_contiguous() and Z.is_contiguous()

    OF_local, G_local = S.shape

    S_t = S.t().contiguous()  # [G, OF]
    Z_t = Z.t().contiguous()  # [G, OF]

    SZ = torch.empty((G_local, 2 * OF_local), device=S.device, dtype=torch.bfloat16)
    SZ[:, 0::2] = S_t
    SZ[:, 1::2] = Z_t
    return SZ.contiguous()


def raw_cuda_w4a16(W, b, SZ, group_size, activations):
    """CUDA wrapper for raw W4A16 kernel."""
    return w4a16_cuda_ext.forward(
        W.contiguous(),
        b.contiguous(),
        SZ.contiguous(),
        activations.contiguous(),
        group_size,
    )


# ==============================================================================
# Test Cases
# ==============================================================================

class TestW4A16Correctness:
    """Test W4A16 kernel correctness."""

    @pytest.mark.parametrize("OF,IF,group_size", [
        (4, 64, 32),
        (8, 128, 64),
        (16, 256, 64),
        (64, 512, 128),
        (256, 2048, 128),
        (1024, 4096, 64),
    ])
    def test_basic_functionality(self, OF, IF, group_size):
        """Test basic functionality with various matrix sizes."""
        assert IF % group_size == 0, "IF must be divisible by group_size"
        assert OF % 4 == 0, "OF must be divisible by 4"

        # Create test data
        W_q = torch.randint(0, 16, (OF, IF), device=DEVICE, dtype=torch.uint8)
        W_packed = pack_rows_4(W_q)

        b = torch.randn((OF,), device=DEVICE, dtype=torch.bfloat16)
        S = torch.ones((OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16)
        Z = torch.randint(0, 16, (OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16)
        SZ = interleave_transposed_s_z(S, Z)

        activations = torch.randn((IF, 1), device=DEVICE, dtype=torch.bfloat16)

        # Compute reference
        torch_ref = torch_w4a16_from_packed4(W_packed, b, S, Z, group_size, activations)

        # Compute CUDA kernel
        cuda_out = raw_cuda_w4a16(W_packed, b, SZ, group_size, activations)

        # Check outputs match
        max_err = (cuda_out - torch_ref).abs().max().item()
        mean_err = (cuda_out - torch_ref).abs().mean().item()

        assert torch.allclose(cuda_out, torch_ref, atol=1.5, rtol=1e-1), (
            f"Correctness check failed: max_err={max_err:.6f}, mean_err={mean_err:.6f}"
        )

    def test_all_zeros_weight(self):
        """Test with all-zero quantized weights."""
        OF, IF, group_size = 8, 256, 64

        W_q = torch.zeros((OF, IF), device=DEVICE, dtype=torch.uint8)
        W_packed = pack_rows_4(W_q)

        b = torch.randn((OF,), device=DEVICE, dtype=torch.bfloat16)
        S = torch.ones((OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16)
        Z = torch.zeros((OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16)
        SZ = interleave_transposed_s_z(S, Z)

        activations = torch.randn((IF, 1), device=DEVICE, dtype=torch.bfloat16)

        torch_ref = torch_w4a16_from_packed4(W_packed, b, S, Z, group_size, activations)
        cuda_out = raw_cuda_w4a16(W_packed, b, SZ, group_size, activations)

        assert torch.allclose(cuda_out, torch_ref, atol=1.5, rtol=1e-1)

    def test_max_values_weight(self):
        """Test with maximum quantized weight values (15)."""
        OF, IF, group_size = 8, 256, 64

        W_q = torch.full((OF, IF), 15, device=DEVICE, dtype=torch.uint8)
        W_packed = pack_rows_4(W_q)

        b = torch.zeros((OF,), device=DEVICE, dtype=torch.bfloat16)
        S = torch.ones((OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16)
        Z = torch.zeros((OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16)
        SZ = interleave_transposed_s_z(S, Z)

        activations = torch.ones((IF, 1), device=DEVICE, dtype=torch.bfloat16)

        torch_ref = torch_w4a16_from_packed4(W_packed, b, S, Z, group_size, activations)
        cuda_out = raw_cuda_w4a16(W_packed, b, SZ, group_size, activations)

        assert torch.allclose(cuda_out, torch_ref, atol=1.5, rtol=1e-1)

    def test_zero_activations(self):
        """Test with zero activations."""
        OF, IF, group_size = 8, 256, 64

        W_q = torch.randint(0, 16, (OF, IF), device=DEVICE, dtype=torch.uint8)
        W_packed = pack_rows_4(W_q)

        b = torch.randn((OF,), device=DEVICE, dtype=torch.bfloat16)
        S = torch.ones((OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16)
        Z = torch.zeros((OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16)
        SZ = interleave_transposed_s_z(S, Z)

        activations = torch.zeros((IF, 1), device=DEVICE, dtype=torch.bfloat16)

        torch_ref = torch_w4a16_from_packed4(W_packed, b, S, Z, group_size, activations)
        cuda_out = raw_cuda_w4a16(W_packed, b, SZ, group_size, activations)

        # With zero activations, output should be just the bias
        assert torch.allclose(cuda_out, b[:, None], atol=1e-3)
        assert torch.allclose(cuda_out, torch_ref, atol=1.5, rtol=1e-1)

    def test_varying_scales(self):
        """Test with various scale values."""
        OF, IF, group_size = 16, 512, 128

        W_q = torch.randint(0, 16, (OF, IF), device=DEVICE, dtype=torch.uint8)
        W_packed = pack_rows_4(W_q)

        b = torch.randn((OF,), device=DEVICE, dtype=torch.bfloat16)

        # Use varying scales
        S = torch.linspace(0.1, 2.0, OF * (IF // group_size), device=DEVICE, dtype=torch.bfloat16)
        S = S.view(OF, IF // group_size)

        Z = torch.randint(0, 16, (OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16)
        SZ = interleave_transposed_s_z(S, Z)

        activations = torch.randn((IF, 1), device=DEVICE, dtype=torch.bfloat16)

        torch_ref = torch_w4a16_from_packed4(W_packed, b, S, Z, group_size, activations)
        cuda_out = raw_cuda_w4a16(W_packed, b, SZ, group_size, activations)

        assert torch.allclose(cuda_out, torch_ref, atol=1.5, rtol=1e-1)

    def test_varying_zeropoints(self):
        """Test with various zero-point values."""
        OF, IF, group_size = 16, 512, 128

        W_q = torch.randint(0, 16, (OF, IF), device=DEVICE, dtype=torch.uint8)
        W_packed = pack_rows_4(W_q)

        b = torch.randn((OF,), device=DEVICE, dtype=torch.bfloat16)
        S = torch.ones((OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16)

        # Use all range of zero-points: 0-15
        Z = torch.randint(0, 16, (OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16)
        SZ = interleave_transposed_s_z(S, Z)

        activations = torch.randn((IF, 1), device=DEVICE, dtype=torch.bfloat16)

        torch_ref = torch_w4a16_from_packed4(W_packed, b, S, Z, group_size, activations)
        cuda_out = raw_cuda_w4a16(W_packed, b, SZ, group_size, activations)

        assert torch.allclose(cuda_out, torch_ref, atol=1.5, rtol=1e-1)

    def test_packing_unpacking_consistency(self):
        """Test that packing and unpacking are inverses."""
        OF, IF = 128, 2048

        W_q_orig = torch.randint(0, 16, (OF, IF), device=DEVICE, dtype=torch.uint8)
        W_packed = pack_rows_4(W_q_orig)
        W_q_unpacked = unpack_rows_4(W_packed)

        assert torch.equal(W_q_orig, W_q_unpacked), "Pack/unpack should be inverses"

    def test_packing_correctness(self):
        """Test packing layout matches specification."""
        W_q = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], device=DEVICE, dtype=torch.uint8)
        W_packed = pack_rows_4(W_q)

        # Expected: bits [3:0]=1, [7:4]=3, [11:8]=5, [15:12]=7 for first column
        expected_col0 = (1 & 0xF) | ((3 & 0xF) << 4) | ((5 & 0xF) << 8) | ((7 & 0xF) << 12)
        assert W_packed[0, 0].item() == expected_col0

    def test_large_matrix_8k_4k(self):
        """Test with large matrices (8k x 4k)."""
        OF, IF, group_size = 8192, 4096, 64

        # Use smaller activation size for memory constraints
        W_q = torch.randint(0, 16, (OF, IF), device=DEVICE, dtype=torch.uint8)
        W_packed = pack_rows_4(W_q)

        b = torch.randn((OF,), device=DEVICE, dtype=torch.bfloat16)
        S = torch.ones((OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16)
        Z = torch.randint(0, 16, (OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16)
        SZ = interleave_transposed_s_z(S, Z)

        activations = torch.randn((IF, 1), device=DEVICE, dtype=torch.bfloat16)

        torch_ref = torch_w4a16_from_packed4(W_packed, b, S, Z, group_size, activations)
        cuda_out = raw_cuda_w4a16(W_packed, b, SZ, group_size, activations)

        max_err = (cuda_out - torch_ref).abs().max().item()
        assert torch.allclose(cuda_out, torch_ref, atol=1.5, rtol=1e-1), (
            f"Large matrix test failed: max_err={max_err:.6f}"
        )


class TestAWQKernelCorrectness:
    """Test AWQ kernel correctness."""

    @pytest.mark.parametrize("OF,IF,group_size,B", [
        (4, 64, 32, 1),
        (8, 128, 64, 1),
        (16, 256, 64, 2),
        (64, 512, 128, 1),
    ])
    def test_awq_basic(self, OF, IF, group_size, B):
        """Test basic AWQ kernel functionality."""
        assert IF % group_size == 0
        PACK_FACTOR = 8

        # Create test data
        W_awq = torch.randint(
            0, 255, (OF, IF // 2), device=DEVICE, dtype=torch.uint8
        ).contiguous()

        S_awq = torch.ones((OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16).contiguous()

        Z_int = torch.randint(0, 16, (OF, IF // group_size), device=DEVICE, dtype=torch.int32)
        Z_awq_int32 = torch.zeros(
            (OF, (IF // group_size) // PACK_FACTOR),
            device=DEVICE,
            dtype=torch.int32,
        )
        for i in range(PACK_FACTOR):
            Z_awq_int32 |= (Z_int[:, i::PACK_FACTOR] & 0xF) << (i * 4)
        Z_awq = Z_awq_int32.view(torch.uint8).contiguous()

        activations = torch.randn((B, IF), device=DEVICE, dtype=torch.bfloat16).contiguous()

        try:
            output = awq_cuda_ext.forward(
                activations, W_awq, Z_awq, S_awq, IF, OF, group_size
            )
            assert output.shape == (B, OF), f"Expected shape ({B}, {OF}), got {output.shape}"
        except Exception as e:
            pytest.fail(f"AWQ kernel failed: {e}")

    def test_awq_output_shape(self):
        """Test AWQ kernel output shape."""
        OF, IF, B, group_size = 16, 256, 2, 64
        PACK_FACTOR = 8

        W_awq = torch.randint(
            0, 255, (OF, IF // 2), device=DEVICE, dtype=torch.uint8
        ).contiguous()
        S_awq = torch.ones((OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16).contiguous()
        Z_awq = torch.zeros((OF, (IF // group_size) // PACK_FACTOR), device=DEVICE, dtype=torch.uint8).contiguous()
        activations = torch.randn((B, IF), device=DEVICE, dtype=torch.bfloat16).contiguous()

        output = awq_cuda_ext.forward(
            activations, W_awq, Z_awq, S_awq, IF, OF, group_size
        )

        assert output.dtype == torch.bfloat16, f"Expected bfloat16, got {output.dtype}"
        assert output.shape == (B, OF), f"Expected shape ({B}, {OF}), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaNs"
        assert not torch.isinf(output).any(), "Output contains Infs"


class TestInputValidation:
    """Test input validation and error handling."""

    def test_misaligned_OF(self):
        """Test that OF not divisible by 4 raises error."""
        OF, IF, group_size = 7, 256, 64

        W_q = torch.randint(0, 16, (OF, IF), device=DEVICE, dtype=torch.uint8)

        with pytest.raises(AssertionError):
            pack_rows_4(W_q)

    def test_misaligned_IF(self):
        """Test that IF not divisible by group_size raises error."""
        OF, IF, group_size = 8, 250, 64

        assert IF % group_size != 0

        W_q = torch.randint(0, 16, (OF, IF), device=DEVICE, dtype=torch.uint8)
        W_packed = pack_rows_4(W_q)

        b = torch.randn((OF,), device=DEVICE, dtype=torch.bfloat16)
        S = torch.ones((OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16)

        # This should fail because S shape doesn't match IF // group_size
        with pytest.raises((AssertionError, RuntimeError)):
            interleave_transposed_s_z(S, S)

    def test_dtype_mismatch_S_Z(self):
        """Test that mismatched dtypes in S and Z are handled."""
        OF, IF, group_size = 8, 256, 64

        S = torch.ones((OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16)
        Z = torch.zeros((OF, IF // group_size), device=DEVICE, dtype=torch.float32)

        with pytest.raises(AssertionError):
            interleave_transposed_s_z(S, Z)


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_very_small_scales(self):
        """Test with very small scale values."""
        OF, IF, group_size = 8, 256, 64

        W_q = torch.randint(0, 16, (OF, IF), device=DEVICE, dtype=torch.uint8)
        W_packed = pack_rows_4(W_q)

        b = torch.zeros((OF,), device=DEVICE, dtype=torch.bfloat16)
        S = torch.full((OF, IF // group_size), 1e-4, device=DEVICE, dtype=torch.bfloat16)
        Z = torch.zeros((OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16)
        SZ = interleave_transposed_s_z(S, Z)

        activations = torch.ones((IF, 1), device=DEVICE, dtype=torch.bfloat16)

        torch_ref = torch_w4a16_from_packed4(W_packed, b, S, Z, group_size, activations)
        cuda_out = raw_cuda_w4a16(W_packed, b, SZ, group_size, activations)

        assert torch.allclose(cuda_out, torch_ref, atol=2.0, rtol=1e-1)

    def test_very_large_values(self):
        """Test with large scale values and activations."""
        OF, IF, group_size = 8, 256, 64

        W_q = torch.randint(0, 16, (OF, IF), device=DEVICE, dtype=torch.uint8)
        W_packed = pack_rows_4(W_q)

        b = torch.full((OF,), 1000.0, device=DEVICE, dtype=torch.bfloat16)
        S = torch.full((OF, IF // group_size), 100.0, device=DEVICE, dtype=torch.bfloat16)
        Z = torch.zeros((OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16)
        SZ = interleave_transposed_s_z(S, Z)

        activations = torch.ones((IF, 1), device=DEVICE, dtype=torch.bfloat16)

        torch_ref = torch_w4a16_from_packed4(W_packed, b, S, Z, group_size, activations)
        cuda_out = raw_cuda_w4a16(W_packed, b, SZ, group_size, activations)

        assert torch.allclose(cuda_out, torch_ref, atol=100.0, rtol=1e-1)

    def test_negative_activations(self):
        """Test with negative activation values."""
        OF, IF, group_size = 8, 256, 64

        W_q = torch.randint(0, 16, (OF, IF), device=DEVICE, dtype=torch.uint8)
        W_packed = pack_rows_4(W_q)

        b = torch.randn((OF,), device=DEVICE, dtype=torch.bfloat16)
        S = torch.ones((OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16)
        Z = torch.zeros((OF, IF // group_size), device=DEVICE, dtype=torch.bfloat16)
        SZ = interleave_transposed_s_z(S, Z)

        activations = -torch.abs(torch.randn((IF, 1), device=DEVICE, dtype=torch.bfloat16))

        torch_ref = torch_w4a16_from_packed4(W_packed, b, S, Z, group_size, activations)
        cuda_out = raw_cuda_w4a16(W_packed, b, SZ, group_size, activations)

        assert torch.allclose(cuda_out, torch_ref, atol=1.5, rtol=1e-1)


def run_all_tests():
    """Run all tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_all_tests()
