import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from datetime import datetime
import argparse

import torch
DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"



########## ALGORITHM (1) ###########

autotune_configs_iep =[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_D': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_D': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_D': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_D': 64, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_D': 128, 'num_stages': 5, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_D': 128, 'num_stages': 5, 'num_warps': 4}),
    ]

@triton.autotune(configs = autotune_configs_iep, key=['N', 'D', 'V'])
@triton.jit
def _indexed_essential_probs_kernel(
    e_ptr, i_ptr, # contains the index of the correct label
    c_ptr, output_ptr,
    N, D, V,
    stride_ed, stride_en,
    stride_cv, stride_cd,
    stride_in, stride_on,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    # Auto-tuned but want to use
    num_stages: tl.constexpr,
):
    # Basically here we index into the Classifier = C, and using the embeddings of the i'th token, E_i
    # We calculate C_{x_i} * E_i
    # I divided E,C into blocks of N, these are for PIDs, then i divide within them into blocks of D
    # Which I iterate over to calculate the dot product of  C_{x_i}, E_i
    pid = tl.program_id(axis=0)
    # I is of shape (N),
    n_dim_offset = pid * BLOCK_SIZE_N # OFFSET OF DIM N
    n_dim_offsets = (n_dim_offset + tl.arange(0, BLOCK_SIZE_N))
    mask_n = n_dim_offsets < N
    c_idxs = tl.load(i_ptr + n_dim_offsets * stride_in, mask=mask_n) # shape of (BLOCK_SIZE_N,)
    # these indices are according to a tensor C with shape (V, D) 
    # Thus, need to be multiplied by stride_cv
    c_offsets = (c_idxs * stride_cv)[:, None] + (tl.arange(0, BLOCK_SIZE_D) * stride_cd)[None, :]
    # Offsets of shape (BLOCK_SIZE_N, BLOCK_SIZE_D)
    e_offsets = (tl.arange(0, BLOCK_SIZE_D) * stride_ed)[:, None] + (n_dim_offsets * stride_en)[None, :]
    # Offsets of shape (BLOCK_SIZE_D, BLOCK_SIZE_N)
    mask_vocab = c_idxs < N
    mask_d = tl.arange(0, BLOCK_SIZE_D)

    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for d in tl.range(0, D, BLOCK_SIZE_D, num_stages=num_stages):
        # Loading C_BLOCK
        mask_c = (mask_vocab)[:, None] & (mask_d < D)[None, :]
        C_BLOCK = tl.load(c_ptr + c_offsets, mask=mask_c) # BN, BD
        # Loading E_BLOCK
        mask_e = (mask_d < D)[:, None] & (mask_n)[None, :]
        E_BLOCK = tl.load(e_ptr + e_offsets, mask=mask_e) # BD, BN

        acc += tl.sum((C_BLOCK * tl.trans(E_BLOCK)), axis=1)
        # We calclate dot product for every E_i , C_i

        c_offsets += BLOCK_SIZE_D * stride_cd
        e_offsets += BLOCK_SIZE_D * stride_ed
        mask_d += BLOCK_SIZE_D

    tl.store(output_ptr + n_dim_offsets * stride_on, acc, mask=mask_n)


def indexed_essential_probs(E, C, I):
    (D, N) , (V, _) = E.shape, C.shape
    assert E.shape[0] == C.shape[1]
    
    # Make sure everything is on the same device
    device = E.device
    E = E.to(device)
    C = C.to(device)
    I = I.to(device)
    O = torch.empty((N,), device=device)
    
    # Make sure we're operating on contiguous tensors
    E = E.contiguous()
    C = C.contiguous()
    I = I.contiguous()
    O = O.contiguous()
    
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    
    try:
        # Pass pointers in the correct order: e_ptr, i_ptr, c_ptr, output_ptr
        _indexed_essential_probs_kernel[grid](
                            E, I, C, O,
                            N, D, V,
                            E.stride(0), E.stride(1),
                            C.stride(0), C.stride(1),
                            I.stride(0), O.stride(0),
                            )
    except Exception as e:
        print(f"Error in kernel execution: {e}")
        import traceback
        traceback.print_exc()
        
    return O

@torch.compile
def torch_indexed_essential_probs(E, C, I):
    # Vectorized approach: batch matrix-vector multiplication
    return torch.einsum('ij,ij->i', C[I], E.T)

def test_indexed_essential_probs(shapes: tuple, atol=1e-2, rtol=1e-1, device=DEVICE):
    # create input data
    torch.manual_seed(0)
    assert type(shapes) == tuple and len(shapes) == 3
    N, D, V = shapes
    E = torch.randn([D, N], device=device)
    C = torch.randn([V, D], device=device)
    O = torch.empty([N], device=device)
    I = torch.randint(high=V, size=(N,), device=device)
    # run kernel & pytorch reference implementation
    c_tri = indexed_essential_probs(E, C, I)
    c_ref = torch_indexed_essential_probs(E, C, I)
    # compare
    torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
    print("PASSED")

def benchmark_indexed_essential_probs(shapes: tuple, device=DEVICE):
    # create input data
    torch.manual_seed(0)
    assert type(shapes) == tuple and len(shapes) == 3
    N, D, V = shapes
    E = torch.randn([D, N], device=device)
    C = torch.randn([V, D], device=device)
    I = torch.randint(high=V, size=(N,), device=device)
    
    # Ensure all tensors are on the correct device and contiguous
    E = E.contiguous().to(device)
    C = C.contiguous().to(device)
    I = I.contiguous().to(device)
    
    # Run the benchmark
    quantiles = [0.5, 0.05, 0.95]
    tri_ms, tri_min_ms, tri_max_ms = triton.testing.do_bench(
        lambda: indexed_essential_probs(E, C, I), 
        quantiles=quantiles
    )
    
    torch_ms, torch_min_ms, torch_max_ms = triton.testing.do_bench(
        lambda: torch_indexed_essential_probs(E, C, I), 
        quantiles=quantiles
    )
    
    print(f"Shape: N={N}, D={D}, V={V}")
    print(f"Triton: {tri_ms:.3f}ms (min: {tri_min_ms:.3f}ms, max: {tri_max_ms:.3f}ms)")
    print(f"PyTorch: {torch_ms:.3f}ms (min: {torch_min_ms:.3f}ms, max: {torch_max_ms:.3f}ms)")
    
    speedup = torch_ms / tri_ms
    print(f"Speedup vs PyTorch: {speedup:.2f}x")
    
    return {
        'triton': (tri_ms, tri_min_ms, tri_max_ms),
        'torch': (torch_ms, torch_min_ms, torch_max_ms)
    }


########## ALGORITHM (2) ###########
# Log-Sum-Exp ( Matrix-Multiplication )

@triton.autotune(configs = autotune_configs_lsemm, key=['N', 'D', 'V'])
@triton.jit
def _lsemm_kernel(
    e_ptr, c_ptr, output_ptr, locks_ptr,
    N, D, V, L,
    stride_ed, stride_en,
    stride_cv, stride_cd,
    stride_on, stride_ll,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_V: tl.constexpr,
    GROUP_SIZE: tl.constexpr, 
):
    # We want to matmul (V, D) @ (D, N)


    PID = tl.program_id(axis=0) 
    
    # Group-major ordering
    num_PID_along_M = tl.cdiv(V, BLOCK_SIZE_V)
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N)
    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    group_id = PID // num_PID_in_group 
    first_PID_in_group_along_M = group_id * GROUP_SIZE 
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE) 
    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
    PID_N = (PID % num_PID_in_group) // group_size_adj

    offsets_V = PID_M * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_D = tl.arange(0, BLOCK_SIZE_D)
    
    # Reference
    offsets_O = offsets_N
    
    
    a_offsets = offsets_V[:, None] * stride_cv + offsets_D[None, :] * stride_cd # (BV, BD)
    b_offsets = offsets_D[:, None] * stride_ed + offsets_N[None, :] * stride_en # (BD, BN)

    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    mx1, mx2 = tl.zeros((1,), dtype=tl.float32), tl.zeros((1,), dtype=tl.float32)
        
    for d in range(0, tl.cdiv(D, BLOCK_SIZE_D)):
        mask = offsets_D < D - d * BLOCK_SIZE_D

        a = tl.load(c_ptr + a_offsets, mask=mask[None, :], other=0.0)
        b = tl.load(e_ptr + b_offsets, mask=mask[:, None], other=0.0)
        
        # a @ b => (BV, BN) and we need to sum over BV

        # Given a maximum of a two matrices, their largest entry in their
        # matmul is at most shared_dim * max1 * max2
        # let's heuristically/randomly assume that for numerical stability and 
        # there will be no problem of gradients, but the LR or something might need to compensate
        # for the irregular order of the graidents.

        guess_factor = 4
        mx1, mx2 = tl.max(a), tl.max(b)

        acc += tl.sum(tl.exp(tl.dot(a, b) - (mx1 * mx2 * (BLOCK_SIZE_D / guess_factor)), axis=0))

        a_offsets += BLOCK_SIZE_D * stride_cd
        b_offsets += BLOCK_SIZE_D * stride_ed

    acc = tl.log(acc)
    
    ointermediate_ptrs = output_ptr + offsets_O * stride_on
    mask_o = (offsets_O < N)

    locks_ptr += PID_N * BLOCK_SIZE_N * stride_ll
    count_ptr = locks_ptr + L * num_PID_along_N * stride_ll

    while tl.atomic_cas(locks_ptr, 0, 1) == 1:
        pass

    count = tl.load(count_ptr)
    if count == 0:
        tl.atomic_xchg(count_ptr, 1)
    else:
        acc += tl.load(ointermediate_ptrs, mask=mask_o)
    
    tl.store(ointermediate_ptrs, acc, mask=mask_o)
    tl.atomic_xchg(locks_ptr, 0)


######### Step 2 #########
def matmul(a, b):
    # check constraints
    assert a.ndim == b.ndim == 2, "only supports matrices, not vectors or tensors"
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    #assert a.is_contiguous() and b.is_contiguous, "input matrices must be contiguous"
    a, b = a.to(torch.float16), b.to(torch.float16)
    
    # get dimesion lengths
    (M, K), (_, N) = a.shape, b.shape
    locks = torch.zeros(2 * meta['BLOCK_SIZE_N'], dtype=torch.int32, device=DEVICE)

    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # cdiv(x, y) = (x + (y - 1)) // y
    # A naive (slow) launch grid might try to separate our axes of parallelizatio into 2 dimensions, one
    #  for cdiv(M, BLOCK_SIZE_M) and the other for cdiv(N, BLOCK_SIZE_N)
    # Here instead we use a 1D launch kernel defined by cdiv(M, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N)
    # The reasoning behind this is explained inside the kernel
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

######### Step 1 #########
def test_matmul_kernel(size: tuple, atol=1e-2, rtol=1e-1, device=DEVICE): # TODO does rtol=0 mean we don't use rtol?
    # create input data
    torch.manual_seed(0)
    assert type(size) == tuple and len(size) == 2
    a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    # run kernel & pytorch reference implementation
    c_tri = matmul(a, b)
    c_ref = torch.matmul(a, b)
    # compare
    torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
    print("PASSED")
























# Add more comprehensive benchmarking with triton.testing.perf_report
configs = [
    triton.testing.Benchmark(
        x_names=["N"],  # We'll vary sequence length
        x_vals=[128 * i for i in range(1, 17)],  # From 128 to 2048
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="Execution Time (ms)",
        plot_name="apple-cce-performance-N",
        args={"D": 512, "V": 2048},  # Fixed dimensions
    ),
    triton.testing.Benchmark(
        x_names=["D"],  # We'll vary embedding dimension
        x_vals=[128 * i for i in range(1, 9)],  # From 128 to 1024
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="Execution Time (ms)",
        plot_name="apple-cce-performance-D",
        args={"N": 512, "V": 2048},  # Fixed dimensions
    ),
    triton.testing.Benchmark(
        x_names=["V"],  # We'll vary vocabulary size
        x_vals=[1000 * i for i in range(1, 11)],  # From 1000 to 10000
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="Execution Time (ms)",
        plot_name="apple-cce-performance-V",
        args={"N": 512, "D": 512},  # Fixed dimensions
    ),
]


@triton.testing.perf_report(configs)
def benchmark(N, D, V, provider):
    # Create input tensors
    torch.manual_seed(0)
    E = torch.randn([D, N], device=DEVICE).contiguous()
    C = torch.randn([V, D], device=DEVICE).contiguous()
    I = torch.randint(high=V, size=(N,), device=DEVICE).contiguous()
    
    quantiles = [0.5, 0.05, 0.95]
    
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: indexed_essential_probs(E, C, I), 
            quantiles=quantiles
        )
    elif provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_indexed_essential_probs(E, C, I), 
            quantiles=quantiles
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    return ms, max_ms, min_ms


def export_results_to_csv(results, filename):
    """Export benchmark results to CSV file"""
    os.makedirs('results', exist_ok=True)
    filepath = os.path.join('results', filename)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Ensure numeric columns are stored as numbers when possible
    numeric_cols = ['N', 'D', 'V']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
    return df


def create_summary_table(df, title, filename):
    """Create a summary table as PNG using matplotlib"""
    # Create figure and axis
    fig = plt.figure(figsize=(12, len(df) * 0.5 + 2), dpi=150)
    ax = plt.subplot(111)
    
    # Hide axes
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Set title
    plt.title(title, fontsize=14, pad=20)
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    filepath = os.path.join('results', filename)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"Table saved to {filepath}")


def run_basic_benchmarks():
    print("Running basic benchmarks...")
    shapes = [
        (128, 128, 128),   # Small
        (256, 256, 256),   # Medium
        (512, 512, 512),   # Large
        (1024, 256, 1024), # Wide sequence
        (256, 1024, 1024), # High dimension
    ]
    
    # Prepare results dictionary
    results = {
        'Shape': [],
        'N': [], 'D': [], 'V': [],
        'Triton (ms)': [], 'PyTorch (ms)': [],
        'Speedup': []
    }
    
    for shape in shapes:
        N, D, V = shape
        shape_str = f"N={N}, D={D}, V={V}"
        print(f"\nBenchmarking shape: {shape_str}")
        timings = benchmark_indexed_essential_probs(shape)
        
        # Extract timing values
        tri_ms = timings['triton'][0]
        torch_ms = timings['torch'][0]
        
        # Calculate speedup
        speedup = torch_ms / tri_ms
        
        # Store results
        results['Shape'].append(shape_str)
        results['N'].append(N)
        results['D'].append(D)
        results['V'].append(V)
        results['Triton (ms)'].append(f"{tri_ms:.2f}")
        results['PyTorch (ms)'].append(f"{torch_ms:.2f}")
        results['Speedup'].append(f"{speedup:.2f}x")
    
    # Export results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"apple_cce_basic_benchmarks_{timestamp}.csv"
    df = export_results_to_csv(results, csv_filename)
    
    # Create summary table as PNG
    png_filename = f"apple_cce_basic_benchmarks_{timestamp}.png"
    create_summary_table(df, "Apple CCE Basic Benchmarks", png_filename)
    
    # Print summary
    print("\n----- BENCHMARK SUMMARY -----")
    for i, shape in enumerate(shapes):
        N, D, V = shape
        print(f"Shape ({results['Shape'][i]}):")
        print(f"  Triton: {results['Triton (ms)'][i]}ms")
        print(f"  PyTorch: {results['PyTorch (ms)'][i]}ms (speedup: {results['Speedup'][i]})")


def run_detailed_benchmarks(show_plots=False):
    """Run detailed benchmarks and manually create plots with matplotlib"""
    print("\nRunning detailed benchmarks...")
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Manual benchmark data collection for each configuration
    # Rather than relying on benchmark.run() which isn't returning data properly
    results = {}
    
    for config in configs:
        config_name = config.plot_name
        x_name = config_name.split('-')[-1]
        x_vals = config.x_vals
        
        print(f"\nBenchmarking with varying {x_name}...")
        
        # Initialize data containers
        data = {
            'triton': [],
            'torch': []
        }
        
        # Collect data for CSV
        csv_data = {
            x_name: [],
            'Triton (ms)': [],
            'PyTorch (ms)': [],
            'Speedup': []
        }
        
        # Run benchmarks for each x value
        for x_val in x_vals:
            print(f"  Running with {x_name}={x_val}...")
            # Set parameters based on config
            params = {**config.args}
            if x_name == 'N':
                params['N'] = x_val
            elif x_name == 'D':
                params['D'] = x_val
            elif x_name == 'V':
                params['V'] = x_val
            
            # Create input tensors
            torch.manual_seed(0)
            E = torch.randn([params['D'], params['N']], device=DEVICE).contiguous()
            C = torch.randn([params['V'], params['D']], device=DEVICE).contiguous()
            I = torch.randint(high=params['V'], size=(params['N'],), device=DEVICE).contiguous()
            
            # Benchmark each provider
            quantiles = [0.5, 0.05, 0.95]
            
            # Triton implementation
            tri_ms, tri_min_ms, tri_max_ms = triton.testing.do_bench(
                lambda: indexed_essential_probs(E, C, I), 
                quantiles=quantiles
            )
            
            # PyTorch
            torch_ms, torch_min_ms, torch_max_ms = triton.testing.do_bench(
                lambda: torch_indexed_essential_probs(E, C, I), 
                quantiles=quantiles
            )
            
            # Store results
            data['triton'].append((x_val, tri_ms))
            data['torch'].append((x_val, torch_ms))
            
            # Store CSV data
            csv_data[x_name].append(x_val)
            csv_data['Triton (ms)'].append(f"{tri_ms * 1000:.2f}")
            csv_data['PyTorch (ms)'].append(f"{torch_ms * 1000:.2f}")
            csv_data['Speedup'].append(f"{torch_ms/tri_ms:.2f}x")
        
        # Save results for this config
        results[config_name] = data
        
        # Export to CSV
        csv_filename = f"apple_cce_detailed_{x_name}_{timestamp}.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(os.path.join('results', csv_filename), index=False)
        print(f"Detailed results for {x_name} saved to results/{csv_filename}")
        
        # Create summary table
        png_filename = f"apple_cce_detailed_{x_name}_{timestamp}.png"
        create_summary_table(df, f"Apple CCE Performance (Varying {x_name})", png_filename)
        
        # Create plots
        create_matplotlib_plots(data, x_name, timestamp)
    
    return results


def create_matplotlib_plots(data, x_name, timestamp):
    """Create performance and speedup plots using matplotlib"""
    # Extract data
    x_vals = [point[0] for point in data['triton']]
    triton_times = [point[1] * 1000 for point in data['triton']]  # Convert to ms
    torch_times = [point[1] * 1000 for point in data['torch']]
    
    # Create performance plot
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(x_vals, triton_times, 'ro-', label='Triton', linewidth=2)
    plt.plot(x_vals, torch_times, 'bo-', label='PyTorch', linewidth=2)
    
    # Add labels and title
    plt.xlabel(x_name, fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title(f'Apple CCE Performance (Varying {x_name})', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Save plot
    os.makedirs('results', exist_ok=True)
    performance_filename = f"apple_cce_performance_{x_name}_{timestamp}.png"
    plt.savefig(os.path.join('results', performance_filename), bbox_inches='tight')
    plt.close()
    print(f"Performance plot saved to results/{performance_filename}")
    
    # Create speedup plot
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Calculate speedups
    speedups = [torch/tri for torch, tri in zip(torch_times, triton_times)]
    
    # Plot speedups
    plt.plot(x_vals, speedups, 'bo-', label='vs PyTorch', linewidth=2)
    
    # Add reference line at y=1
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3, label='Baseline (Equal Performance)')
    
    # Add labels and title
    plt.xlabel(x_name, fontsize=12)
    plt.ylabel('Speedup (x times)', fontsize=12)
    plt.title(f'Triton Speedup (Varying {x_name})', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Save speedup plot
    speedup_filename = f"apple_cce_speedup_{x_name}_{timestamp}.png"
    plt.savefig(os.path.join('results', speedup_filename), bbox_inches='tight')
    plt.close()
    print(f"Speedup plot saved to results/{speedup_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Apple CCE benchmarks')
    parser.add_argument('--test-only', action='store_true', help='Run only the correctness test')
    parser.add_argument('--basic-benchmarks', action='store_true', help='Run basic benchmarks')
    parser.add_argument('--detailed-benchmarks', action='store_true', help='Run detailed benchmarks with plots')
    parser.add_argument('--all', action='store_true', help='Run all tests and benchmarks')
    
    args = parser.parse_args()
    
    # Default to running all if no arguments are provided
    run_all = args.all or (not args.test_only and not args.basic_benchmarks and not args.detailed_benchmarks)
    
    if torch.cuda.is_available():
        print("CUDA is available, running on GPU")
    else:
        print("CUDA is not available, running on CPU")
    
    try:
        # Always run the correctness test first
        print("Running correctness test...")
        test_indexed_essential_probs(shapes=(128, 128, 128))
        print("Test passed!")
        
        if args.basic_benchmarks or run_all:
            print("\nRunning basic benchmarks...")
            os.makedirs('results', exist_ok=True)
            run_basic_benchmarks()
        
        if args.detailed_benchmarks or run_all:
            # Use our new implementation that doesn't rely on benchmark.run()
            benchmark_data = run_detailed_benchmarks(show_plots=False)
            print("\nAll benchmarks complete. Results saved to the 'results' directory.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()