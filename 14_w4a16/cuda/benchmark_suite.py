import torch
from torch.utils.cpp_extension import load
import os

os.environ["PATH"] += r";C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64"
os.environ["PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin" + ";" + os.environ["PATH"]
# Use the root directory of Nsight Compute
NSIGHT_PATH = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.4.1"
os.environ["PATH"] = NSIGHT_PATH + ";" + os.path.join(NSIGHT_PATH, "target", "windows-desktop-win7-x64") + ";" + os.environ["PATH"]

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from datetime import datetime

this_dir = os.path.dirname(os.path.abspath(__file__))

w4a16_cuda_ext = load(
    name="w4a16_cuda_ext",
    sources=[
        os.path.join(this_dir, "w4a16_cuda.cu"),
    ],
    extra_cuda_cflags=["-O3", "-allow-unsupported-compiler"],
    verbose=True,
)
import nsight

DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"


def raw_cuda_w4a16(W, b, S, Z, group_size, activations):
    W = W.contiguous()
    b = b.contiguous()
    S = S.contiguous()
    Z = Z.contiguous()
    activations = activations.contiguous()
    return w4a16_cuda_ext.forward(W, b, S, Z, activations, group_size)


# ==========================================================
# PyTorch Reference
# ==========================================================

def unpacking_layer(W_qp: torch.Tensor):
    w0 = W_qp & 0x0F
    w1 = (W_qp >> 4) & 0x0F
    return torch.stack([w0, w1], dim=1).view(-1, W_qp.shape[1])


def dequantize_layer(W_q: torch.Tensor, S: torch.Tensor, Z: torch.Tensor, b_q: torch.Tensor, group_size: int):
    N, K = W_q.shape
    W_q_reshaped = W_q.view(N, K // group_size, group_size)
    S_expanded = S.unsqueeze(-1)
    Z_expanded = Z.unsqueeze(-1)

    W_deq_reshaped = (W_q_reshaped.to(torch.float16) - Z_expanded) * S_expanded
    return W_deq_reshaped.view(N, K), b_q


def torch_w4a16(W, b, S, Z, group_size, activations):
    activations = activations.to(torch.float16)
    W_unpacked = unpacking_layer(W)
    W_deq, b_deq = dequantize_layer(W_unpacked, S, Z, b, group_size)

    W_deq = W_deq.to(torch.float16)
    out = torch.matmul(W_deq, activations)

    if b_deq is not None:
        out += b_deq[:, None]

    return out


# ==========================================================
# Benchmark Helper
# ==========================================================

def simple_bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# ==========================================================
# Testing
# ==========================================================

def test_w4a16(shapes, atol=1e-2, rtol=1e-1, device=DEVICE):

    torch.manual_seed(0)

    OF_actual, IF, B = shapes
    OF_packed = OF_actual // 2
    group_size = 16

    W_packed = torch.randint(0, 256, (OF_packed, IF), device=device, dtype=torch.uint8)
    b = torch.randn((OF_actual,), device=device, dtype=torch.float16)

    L = max(64, IF // group_size)

    S = torch.ones((OF_actual, L), device=device, dtype=torch.float16)
    Z = torch.zeros_like(S)

    activations = torch.randn((IF, B), device=device, dtype=torch.float16)

    out_ref = torch_w4a16(W_packed, b, S, Z, group_size, activations)
    out_cuda = raw_cuda_w4a16(W_packed, b, S, Z, group_size, activations)

    torch.testing.assert_close(out_cuda, out_ref, atol=atol, rtol=rtol)

    print("w4a16 unit test PASSED")


# ==========================================================
# Benchmarking
# ==========================================================

def benchmark_w4a16(shapes, device=DEVICE):

    torch.manual_seed(0)

    OF_actual, IF, B = shapes
    OF_packed = OF_actual // 2
    group_size = 128

    W_packed = torch.randint(0, 256, (OF_packed, IF), device=device, dtype=torch.uint8).contiguous()
    b = torch.randn((OF_actual,), device=device, dtype=torch.float16).contiguous()

    L = max(64, IF // group_size)

    S = torch.ones((OF_actual, L), device=device, dtype=torch.float16).contiguous()
    Z = torch.zeros_like(S).contiguous()

    activations = torch.randn((IF, B), device=device, dtype=torch.float16).contiguous()

    W_ref_fp16 = torch.randn((OF_actual, IF), device=device, dtype=torch.float16).contiguous()

    cuda_ms = simple_bench(
        lambda: raw_cuda_w4a16(W_packed, b, S, Z, group_size, activations)
    )

    torch_ms = simple_bench(
        lambda: torch.matmul(W_ref_fp16, activations) + b[:, None]
    )

    print(f"Shape: OF={OF_actual}, IF={IF}, B={B}")
    print(f"Raw CUDA W4A16: {cuda_ms:.3f}ms")
    print(f"PyTorch FP16: {torch_ms:.3f}ms")

    return cuda_ms, torch_ms


# ==========================================================
# Plotting
# ==========================================================

def plotting_and_benchmarking():
    OF_vals = [4096, 8192, 16384, 32768]
    IF = 8192
    B = 1

    records = []

    for OF in OF_vals:

        print(f"\nBenchmarking OF={OF}")

        cuda_ms, torch_ms = benchmark_w4a16((OF, IF, B))

        flops = 2.0 * OF * IF * B

        records.append({
            "OF": OF,
            "provider": "raw_cuda",
            "ms": cuda_ms,
            "tflops": flops * 1e-12 / (cuda_ms * 1e-3)
        })

        records.append({
            "OF": OF,
            "provider": "torch",
            "ms": torch_ms,
            "tflops": flops * 1e-12 / (torch_ms * 1e-3)
        })

    df = pd.DataFrame(records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = f"w4a16_benchmark_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    print("Saved CSV:", csv_path)

    plt.figure(figsize=(8,5))

    for provider in df.provider.unique():

        sub = df[df.provider == provider]

        plt.plot(sub.OF, sub.tflops, marker="o", label=provider)

    plt.xlabel("Output features")
    plt.ylabel("TFLOPS")

    plt.grid(True)

    plt.legend()

    img_path = f"w4a16_benchmark_{timestamp}.png"
    plt.savefig(img_path)

    print("Saved plot:", img_path)


# ==========================================================
# Nsight Profiling
# ==========================================================

#@nsight.analyze.plot('w4a16_sweep.png')
@nsight.analyze.kernel(
    configs=[8192,], 
    runs=1,
    metrics=[
        "gpu__time_duration.sum", 
        "sm__throughput.avg.pct_of_peak_sustained_elapsed", # Compute SOL
        "dram__throughput.avg.pct_of_peak_sustained_elapsed" # Memory SOL
    ],
    # clock_control='base' can help with the 'frequencies' warning
    clock_control='base',
)
def run_nsight_benchmark_profiling(OF_actual: int) -> None:
    IF = 8192
    B = 1
    device = DEVICE
    group_size = 16
    OF_packed = OF_actual // 2
    L = max(64, IF // group_size)

    W_packed = torch.randint(0, 256, (OF_packed, IF), device=device, dtype=torch.uint8)
    b = torch.randn((OF_actual,), device=device, dtype=torch.float16)
    S = torch.ones((OF_actual, L), device=device, dtype=torch.float16)
    Z = torch.zeros_like(S)
    activations = torch.randn((IF, B), device=device, dtype=torch.float16)

    # 1. Warmup - IMPORTANT
    # This ensures the GPU is awake and the kernel is cached
    raw_cuda_w4a16(W_packed, b, S, Z, group_size, activations)
    torch.cuda.synchronize()

    # 2. The Profiled Call
    # Use a unique name for each config to help the extractor
    with nsight.annotate(f"run_of_{OF_actual}"):
        out = raw_cuda_w4a16(W_packed, b, S, Z, group_size, activations)
    
    # 3. Force completion
    torch.cuda.synchronize()
    # Adding a small dummy operation ensures the 'out' tensor isn't optimized away
    _ = out.sum().item()
# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("Running correctness test")
    test_w4a16((1024, 1024, 1))

    #print("Running Nsight profiling")
    #run_nsight_benchmark_profiling()

    print("Running benchmarking and plotting")
    plotting_and_benchmarking()