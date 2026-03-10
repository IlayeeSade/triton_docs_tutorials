import torch
from torch.utils.cpp_extension import load

# 1. JIT Compile the CUDA kernel (includes -lineinfo for Nsight Compute)
print("Compiling CUDA kernel... (this takes a few seconds the first time)")
w4a16_cuda_ext = load(
    name="w4a16_cuda_ext",
    sources=["w4a16_kernel.cu"],
    extra_cuda_cflags=['-O3', '-lineinfo'],
    verbose=True
)
print("Compilation finished!")

# 2. Setup dimensions
B = 1
IF = 4096
OF_packed = 4096 // 2
group_size = 128

# 3. Create dummy tensors on the GPU
W_packed = torch.randint(0, 255, (OF_packed, IF // 4), dtype=torch.uint8, device='cuda')
activations = torch.randn((IF, B), dtype=torch.float16, device='cuda')
b = torch.randn(OF_packed * 2, dtype=torch.float16, device='cuda')
S = torch.ones((OF_packed * 2, IF // group_size), dtype=torch.float16, device='cuda')
Z = torch.zeros((OF_packed * 2, IF // group_size), dtype=torch.float16, device='cuda')

# 4. Warm-up
for _ in range(10):
    _ = w4a16_cuda_ext.forward(W_packed, b, S, Z, activations, group_size)
torch.cuda.synchronize()

# 5. Profiled Run
out = w4a16_cuda_ext.forward(W_packed, b, S, Z, activations, group_size)
torch.cuda.synchronize()

print("Kernel executed successfully. Output shape:", out.shape)