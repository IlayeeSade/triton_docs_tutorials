import torch
from torch.utils.cpp_extension import load
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from datetime import datetime

# Ensure the AWQ .cu file is cleaned (no ../matmul.h) before running
this_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Load Extensions
w4a16_cuda_ext = load(
    name="w4a16_cuda_ext",
    sources=[os.path.join(this_dir, "w4a16_cuda.cu")],
    extra_cuda_cflags=["-O3", "-allow-unsupported-compiler"],
    verbose=True,
)

awq_cuda_ext = load(
    name="awq_cuda_ext",
    sources=[os.path.join(this_dir, "awq_kernel.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 2. Packing / Unpacking Helpers
# ==============================================================================
def pack_rows_4(W_q: torch.Tensor) -> torch.Tensor:
    """
    Pack 4 adjacent rows into one 16-bit element:
      bits [3:0]   = row 0
      bits [7:4]   = row 1
      bits [11:8]  = row 2
      bits [15:12] = row 3

    Input:
      W_q: [OF, IF] uint8/int tensor with values in [0, 15]

    Output:
      [OF // 4, IF] uint16
    """
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


def unpack_rows_4(W_packed: torch.Tensor) -> torch.Tensor:
    """
    Inverse of pack_rows_4.

    Input:
      [OF // 4, IF] uint16/int16

    Output:
      [OF, IF] uint8
    """
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


# ==============================================================================
# 3. Kernel Wrapper
# ==============================================================================
def raw_cuda_w4a16(W, b, SZ, group_size, activations):
    # CUDA wrapper expects:
    #   W  : [OF // 4, IF] uint16
    #   b  : [OF] float16
    #   SZ : [IF // group_size, 2 * OF] float16
    #   activations : [IF, 1] float16
    return w4a16_cuda_ext.forward(
        W.contiguous(),
        b.contiguous(),
        SZ.contiguous(),          # FIX: do NOT transpose
        activations.contiguous(),
        group_size,
    )


# ==============================================================================
# 4. PyTorch Reference
# ==============================================================================
def dequantize_layer(W_q, S, Z, group_size):
    N, K = W_q.shape
    W_q_reshaped = W_q.view(N, K // group_size, group_size)
    W_deq = (W_q_reshaped.to(torch.float16) - Z.unsqueeze(-1)) * S.unsqueeze(-1)
    return W_deq.view(N, K)


def torch_w4a16_from_packed4(W_packed, b, S, Z, group_size, activations):
    W_unpacked = unpack_rows_4(W_packed)
    W_deq = dequantize_layer(W_unpacked, S, Z, group_size)
    out = torch.matmul(W_deq, activations) + b[:, None]
    return out


# ==============================================================================
# 5. SZ Layout Helper
# ==============================================================================
def interleave_transposed_s_z(S: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """
    Convert:
      S, Z: [OF, G]
    into:
      SZ: [G, 2 * OF]

    Layout for each group g:
      [s0, z0, s1, z1, s2, z2, ...]
    """
    assert S.shape == Z.shape
    assert S.dtype == torch.float16 and Z.dtype == torch.float16
    assert S.is_contiguous() and Z.is_contiguous()

    OF_local, G_local = S.shape

    S_t = S.t().contiguous()  # [G, OF]
    Z_t = Z.t().contiguous()  # [G, OF]

    SZ = torch.empty((G_local, 2 * OF_local), device=S.device, dtype=torch.float16)
    SZ[:, 0::2] = S_t
    SZ[:, 1::2] = Z_t
    return SZ.contiguous()


# ==============================================================================
# 6. Main Benchmarking Logic
# ==============================================================================
def plotting_and_benchmarking():
    OF_vals = [4096, 8192, 16384, 32768]
    IF = 8192
    B = 1
    group_size = 64
    PACK_FACTOR = 8

    records = []

    for OF in OF_vals:
        print(f"\n--- Benchmarking OF={OF} ---")

        # ----------------------------------------------------------------------
        # Create one logical quantized weight matrix first: [OF, IF], values 0..15
        # Then derive provider-specific packed layouts from it.
        # ----------------------------------------------------------------------
        W_q = torch.randint(0, 16, (OF, IF), device=DEVICE, dtype=torch.uint8).contiguous()

        # Raw CUDA uses 4-row packing
        W_packed_raw = pack_rows_4(W_q)  # [OF // 4, IF], uint16

        b = torch.randn((OF,), device=DEVICE, dtype=torch.float16).contiguous()
        G = IF // group_size

        S = torch.ones((OF, G), device=DEVICE, dtype=torch.float16).contiguous()
        Z = torch.randint(0, 16, (OF, G), device=DEVICE, dtype=torch.float16).contiguous()

        SZ = interleave_transposed_s_z(S, Z)

        act = torch.randn((IF, B), device=DEVICE, dtype=torch.float16).contiguous()
        W_ref_fp16 = torch.randn((OF, IF), device=DEVICE, dtype=torch.float16).contiguous()

        # Sanity checks for the CUDA wrapper contract
        assert W_packed_raw.shape == (OF // 4, IF)
        assert b.shape == (OF,)
        assert SZ.shape == (IF // group_size, 2 * OF)
        assert act.shape == (IF, 1)

        # ----------------------------------------------------------------------
        # Correctness check: raw CUDA vs Torch reference for the 4-row packing
        # ----------------------------------------------------------------------
        torch_ref_raw = torch_w4a16_from_packed4(W_packed_raw, b, S, Z, group_size, act)
        cuda_out = raw_cuda_w4a16(W_packed_raw, b, SZ, group_size, act)

        max_abs_err = (cuda_out - torch_ref_raw).abs().max().item()
        mean_abs_err = (cuda_out - torch_ref_raw).abs().mean().item()

        print(
            f"raw_cuda vs torch(pack4) | "
            f"max abs err = {max_abs_err:.6f}, mean abs err = {mean_abs_err:.6f}"
        )

        assert torch.allclose(cuda_out, torch_ref_raw, atol=1.5, rtol=1e-1), (
            f"raw_cuda correctness check failed for OF={OF}: "
            f"max_abs_err={max_abs_err:.6f}, mean_abs_err={mean_abs_err:.6f}"
        )

        # ----------------------------------------------------------------------
        # Data for AWQ
        # ----------------------------------------------------------------------
        W_awq = torch.randint(
            0, 255, (OF, IF // 2), device=DEVICE, dtype=torch.uint8
        ).contiguous()

        Z_int = Z.to(torch.int32)
        Z_awq_int32 = torch.zeros(
            (OF, (IF // group_size) // PACK_FACTOR),
            device=DEVICE,
            dtype=torch.int32,
        )

        for i in range(PACK_FACTOR):
            Z_awq_int32 |= (Z_int[:, i::PACK_FACTOR] & 0xF) << (i * 4)

        Z_awq = Z_awq_int32.view(torch.uint8).contiguous()
        S_awq = S.clone().contiguous()

        # Transpose activations to [B, IF] for AWQ
        act_awq = act.t().contiguous()

        # ----------------------------------------------------------------------
        # Benchmarking
        # ----------------------------------------------------------------------
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        def bench(fn, iters=100):
            for _ in range(10):
                fn()
            torch.cuda.synchronize()

            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize()

            return start.elapsed_time(end) / iters

        torch_ms = bench(lambda: torch.matmul(W_ref_fp16, act) + b[:, None])
        cuda_ms = bench(lambda: raw_cuda_w4a16(W_packed_raw, b, SZ, group_size, act))

        try:
            awq_ms = bench(
                lambda: awq_cuda_ext.forward(
                    act_awq, W_awq, Z_awq, S_awq, IF, OF, group_size
                )
            )
        except Exception as e:
            print(f"AWQ Error: {e}")
            awq_ms = float("nan")

        flops = 2.0 * OF * IF * B

        for name, ms in [("torch", torch_ms), ("raw_cuda", cuda_ms), ("awq", awq_ms)]:
            if not np.isnan(ms):
                records.append(
                    {
                        "OF": OF,
                        "provider": name,
                        "ms": ms,
                        "tflops": (flops * 1e-12) / (ms * 1e-3),
                    }
                )

    # --------------------------------------------------------------------------
    # Plotting
    # --------------------------------------------------------------------------
    df = pd.DataFrame(records)
    pivot_df = df.pivot(index="OF", columns="provider", values="tflops")

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {"torch": "gray", "raw_cuda": "tab:green", "awq": "tab:red"}

    for provider in df.provider.unique():
        sub = df[df.provider == provider]
        ax.plot(
            sub.OF,
            sub.tflops,
            marker="o",
            label=provider.upper(),
            color=colors.get(provider),
        )

        if provider != "torch":
            for of_val in sub.OF:
                try:
                    speedup = pivot_df.loc[of_val, provider] / pivot_df.loc[of_val, "torch"]
                    ax.annotate(
                        f"{speedup:.2f}x",
                        xy=(of_val, pivot_df.loc[of_val, provider]),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha="center",
                        fontweight="bold",
                    )
                except KeyError:
                    continue

    ax.set_title(f"W4A16 GEMV Performance Comparison (IF={IF}, B={B})")
    ax.set_ylabel("TFLOPS")
    ax.set_xlabel("Output Features")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(f"benchmark_full_{datetime.now().strftime('%H%M%S')}.png")
    plt.show()


if __name__ == "__main__":
    plotting_and_benchmarking()