import torch

from w4a16 import w4a16, DEVICE


def test_w4a16_matmul_correctness(
    shapes: tuple = (256, 512, 128), atol: float = 1e-2, rtol: float = 1e-1, device=DEVICE
):
    """
    Basic correctness test that the matrix multiplication performed inside the
    Triton `w4a16` kernel matches a standard PyTorch matmul for the same
    weights and activations.
    """
    torch.manual_seed(0)
    assert isinstance(shapes, tuple) and len(shapes) == 3
    OF, IF, B = shapes

    # Random FP16 weights and activations for a standard matmul:
    # (OF, IF) @ (IF, B) -> (OF, B)
    W = torch.randn((OF, IF), device=device, dtype=torch.float16)
    activations = torch.randn((IF, B), device=device, dtype=torch.float16)

    # The current Triton kernel signature expects additional tensors (b, S, Z, group_size),
    # but the kernel body does not actually use them. We pass minimal dummy tensors
    # just to satisfy the interface.
    b = torch.zeros((OF, 1), device=device, dtype=torch.float16)
    S = torch.ones((1, 1), device=device, dtype=torch.float16)
    Z = torch.zeros((1, 1), device=device, dtype=torch.float16)
    group_size = torch.ones((1, 1), device=device, dtype=torch.int32)

    out_triton = w4a16(W, b, S, Z, group_size, activations)

    # Reference result using PyTorch matmul on the same FP16 data
    out_ref = torch.matmul(W, activations)

    torch.testing.assert_close(out_triton, out_ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    # Simple manual run hook
    test_w4a16_matmul_correctness()
    print("w4a16 matmul correctness test PASSED")

