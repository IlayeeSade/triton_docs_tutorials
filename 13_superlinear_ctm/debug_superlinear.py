import torch
import triton
import triton.language as tl
import numpy as np

DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

# Simple test case to debug the kernel
def debug_superlinear():
    print("Debugging SuperLinear kernel issues...")
    
    # Small test case
    B, D, M, H = 1, 2, 4, 1  # Very small for debugging
    
    # Create test tensors with known values
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0]]], device=DEVICE, dtype=torch.float32)  # (1, 2, 4)
    
    w1 = torch.tensor([[[0.1, 0.2],    # (4, 1, 2)
                       [0.3, 0.4],
                       [0.5, 0.6],
                       [0.7, 0.8]]], device=DEVICE, dtype=torch.float32)
    
    b1 = torch.tensor([[[0.01],        # (1, 2, 1)
                       [0.02]]], device=DEVICE, dtype=torch.float32)
    
    print(f"Input shapes: x={x.shape}, w1={w1.shape}, b1={b1.shape}")
    print(f"x:\n{x}")
    print(f"w1:\n{w1}")
    print(f"b1:\n{b1}")
    
    # Expected result from einsum
    expected = torch.einsum('BDM,MHD->BDH', x, w1) + b1
    print(f"Expected result (einsum):\n{expected}")
    
    # Let's manually compute what should happen:
    # For B=1, D=2, M=4, H=1
    # einsum('BDM,MHD->BDH') means:
    # result[b,d,h] = sum_m x[b,d,m] * w1[m,h,d]
    
    manual_result = torch.zeros(1, 2, 1, device=DEVICE)
    for b in range(1):
        for d in range(2):
            for h in range(1):
                for m in range(4):
                    manual_result[b,d,h] += x[b,d,m] * w1[m,h,d]
    manual_result += b1
    
    print(f"Manual computation:\n{manual_result}")
    
    # Now let's analyze what the kernel is doing wrong
    print("\n" + "="*50)
    print("ANALYZING KERNEL ISSUES")
    print("="*50)
    
    # Issue 1: Bias loading
    print("\nIssue 1: Bias loading")
    print("Current kernel loads bias as: b1_ptrs = b1_ptr + offs_h * stride_bh + pid_d * stride_bd")
    print("This means for pid_d=0,1: bias[0,0,0] and bias[0,1,0] are loaded")
    print("But bias should be loaded as: bias[0,d,h] for each d,h combination")
    
    # Issue 2: Matrix multiplication indexing
    print("\nIssue 2: Matrix multiplication indexing")
    print("Kernel loads w1 as: w1_ptrs = w1_ptr + (offs_m * stride_wm + offs_h * stride_wh + pid_d * stride_wd)")
    print("This means w1[m,h,d] where d is the program ID")
    print("But for einsum('BDM,MHD->BDH'), we need w1[m,h,d] where d varies")
    
    # Issue 3: Loop structure
    print("\nIssue 3: Loop structure")
    print("The kernel loops over M dimension in blocks, but the einsum operation")
    print("requires a full reduction over the M dimension for each B,D,H combination")
    
    return expected, manual_result

if __name__ == "__main__":
    debug_superlinear() 