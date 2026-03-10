# Debugging Timeline: PyTorch CUDA C++ Extension

At the start, there were several separate problems stacked on top of each other. 

## 1. Missing CUDA Toolkit
First, PyTorch could not find your CUDA Toolkit at all. That was the `CUDA_HOME environment variable is not set` error. 

Your script uses `torch.utils.cpp_extension.load(...)`, and that build path needs to know where CUDA is installed so it can find:
* `nvcc`
* CUDA headers
* CUDA libraries

**Result:** The build failed before compilation really started.

---

## 2. PyTorch CPU-Only Install
After the path was fixed, the next real issue appeared: your PyTorch install was CPU-only. You had:
```python
torch: 2.10.0+cpu
torch.version.cuda: None
torch.cuda.is_available(): False