# CUDA Kernel Profiling with Nsight Compute (Windows)

This guide explains how to profile the CUDA kernel using **NVIDIA Nsight Compute (`ncu`)** when building a **PyTorch CUDA extension** on Windows.

The important requirement is that compilation must happen inside the **Visual Studio C++ environment**, otherwise `nvcc` will not find `cl.exe`.

---

# 1. Open the Correct Terminal

You **must start from**:x64 Native Tools Command Prompt for VS 2022

# 2. Navigate to the CUDA Project

```
cd C:\Users\elais\OneDrive\Desktop\w4a16_proj\triton_docs_tutorials\14_w4a16\cuda
```

# 3. Launch PowerShell (Optional but Recommended)

```
powershell
```

# 4. Allow Virtual Environment Activation

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

# 5. Activate the Python Virtual Environment

```
.\.venv\Scripts\Activate.ps1
```

# 6. Verify the Compiler Exists

```
cl
```

# 7. Run Nsight Compute Profiling

```
ncu -o my_kernel_profile --set full python profile_kernel.py
```