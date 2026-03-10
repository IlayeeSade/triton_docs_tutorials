import os
import subprocess
import torch

# ----- hard-set CUDA for this process -----
CUDA_PATH = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"

os.environ["CUDA_HOME"] = CUDA_PATH
os.environ["CUDA_PATH"] = CUDA_PATH
os.environ["PATH"] = CUDA_PATH + r"\bin;" + CUDA_PATH + r"\libnvvp;" + os.environ["PATH"]

# Optional but often helps the JIT builder find cl.exe faster on Windows
os.environ.setdefault("DISTUTILS_USE_SDK", "1")

import torch.utils.cpp_extension as cpp_ext
from torch.utils.cpp_extension import load

# Force PyTorch's cached CUDA path too
cpp_ext.CUDA_HOME = CUDA_PATH

# ----- diagnostics -----
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("CUDA_HOME env:", os.environ.get("CUDA_HOME"))
print("CUDA_PATH env:", os.environ.get("CUDA_PATH"))
print("cpp_ext.CUDA_HOME:", cpp_ext.CUDA_HOME)

try:
    nvcc_out = subprocess.check_output(["nvcc", "--version"], text=True, stderr=subprocess.STDOUT)
    print("\nNVCC OK:\n", nvcc_out)
except Exception as e:
    print("\nNVCC NOT FOUND OR FAILED:", repr(e))

current_dir = os.path.dirname(os.path.abspath(__file__))
source_file = os.path.join(current_dir, "w4a16_cuda.cu")

print(f"Compiling CUDA kernel from: {source_file}")

w4a16_cuda_ext = load(
    name="w4a16_cuda_ext",
    sources=[source_file],
    extra_cuda_cflags=[
        "-O3",
        "-lineinfo",
        "-Xcompiler", "/Zc:__cplusplus",
        "-Xcompiler", "/std:c++17",
    ],
    extra_cflags=[
        "/std:c++17",
        "/Zc:__cplusplus",
    ],
    verbose=True,
)

print("Compilation finished!")

# 2. Setup dimensions
B = 1
IF = 4096
OF_packed = 4096 // 2
group_size = 128

# 3. Create dummy tensors on the GPU
W_packed = torch.randint(0, 255, (OF_packed, IF // 4), dtype=torch.uint8, device="cuda")
activations = torch.randn((IF, B), dtype=torch.float16, device="cuda")
b = torch.randn(OF_packed * 2, dtype=torch.float16, device="cuda")
S = torch.ones((OF_packed * 2, IF // group_size), dtype=torch.float16, device="cuda")
Z = torch.zeros((OF_packed * 2, IF // group_size), dtype=torch.float16, device="cuda")


def interleave_transposed_s_z(S: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    assert S.shape == Z.shape
    assert S.dtype == torch.float16 and Z.dtype == torch.float16
    assert S.is_contiguous() and Z.is_contiguous()

    # Transpose: [OF, G] -> [G, OF]
    S_t = S.transpose(0, 1).contiguous()
    Z_t = Z.transpose(0, 1).contiguous()

    G, OF = S_t.shape

    # Stack into [G, OF, 2], where last dim is [S, Z]
    SZ = torch.stack((S_t, Z_t), dim=-1)

    # Flatten last two dims: [G, OF, 2] -> [G, 2*OF]
    # Row layout becomes: S_col0, Z_col0, S_col1, Z_col1, ...
    SZ_interleaved = SZ.reshape(G, 2 * OF).contiguous()

    return SZ_interleaved


SZ = interleave_transposed_s_z(S, Z)

# 4. Warm-up
for _ in range(10):
    _ = w4a16_cuda_ext.forward(W_packed, b, SZ, activations, group_size)
torch.cuda.synchronize()

# 5. Profiled Run
out = w4a16_cuda_ext.forward(W_packed, b, SZ, activations, group_size)
torch.cuda.synchronize()

print("Kernel executed successfully. Output shape:", out.shape)