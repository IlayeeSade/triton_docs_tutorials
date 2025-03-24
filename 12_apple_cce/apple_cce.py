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

autotune_configs =[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_D': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_D': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_D': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_D': 64, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_D': 128, 'num_stages': 5, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_D': 128, 'num_stages': 5, 'num_warps': 4}),
    ]

@triton.autotune(configs = autotune_configs, key=['N', 'D', 'V'])
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


def torch_indexed_essential_probs(E, C, I, use_vectorized=True):
    N = I.shape[0]
    if use_vectorized:
        indexed_C = C[I]
        E_t = E.T
        O = (indexed_C * E_t).sum(dim=1)
    else:
        O = torch.empty(N, device=I.device)
        for i in range(N):
            idx = I[i]
            c_row = C[idx]
            # Get the i-th column from E
            e_col = E[:, i]
            O[i] = torch.dot(c_row, e_col)
    return O

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
        lambda: torch_indexed_essential_probs(E, C, I, use_vectorized=True), 
        quantiles=quantiles
    )
    
    torch_non_vec_ms, torch_non_vec_min_ms, torch_non_vec_max_ms = triton.testing.do_bench(
        lambda: torch_indexed_essential_probs(E, C, I, use_vectorized=False), 
        quantiles=quantiles
    )
    
    print(f"Shape: N={N}, D={D}, V={V}")
    print(f"Triton: {tri_ms:.3f}ms (min: {tri_min_ms:.3f}ms, max: {tri_max_ms:.3f}ms)")
    print(f"PyTorch Vectorized: {torch_ms:.3f}ms (min: {torch_min_ms:.3f}ms, max: {torch_max_ms:.3f}ms)")
    print(f"PyTorch Non-Vectorized: {torch_non_vec_ms:.3f}ms (min: {torch_non_vec_min_ms:.3f}ms, max: {torch_non_vec_max_ms:.3f}ms)")
    
    speedup_vs_vec = torch_ms / tri_ms
    speedup_vs_nonvec = torch_non_vec_ms / tri_ms
    print(f"Speedup vs PyTorch Vectorized: {speedup_vs_vec:.2f}x")
    print(f"Speedup vs PyTorch Non-Vectorized: {speedup_vs_nonvec:.2f}x")
    
    return {
        'triton': (tri_ms, tri_min_ms, tri_max_ms),
        'torch_vec': (torch_ms, torch_min_ms, torch_max_ms),
        'torch_non_vec': (torch_non_vec_ms, torch_non_vec_min_ms, torch_non_vec_max_ms)
    }


# Add more comprehensive benchmarking with triton.testing.perf_report
configs = [
    triton.testing.Benchmark(
        x_names=["N"],  # We'll vary sequence length
        x_vals=[128 * i for i in range(1, 17)],  # From 128 to 2048
        line_arg="provider",
        line_vals=["triton", "torch_vec", "torch_non_vec"],
        line_names=["Triton", "PyTorch Vectorized", "PyTorch Non-Vectorized"],
        styles=[("red", "-"), ("blue", "-"), ("green", "--")],
        ylabel="Execution Time (ms)",
        plot_name="apple-cce-performance-N",
        args={"D": 512, "V": 2048},  # Fixed dimensions
    ),
    triton.testing.Benchmark(
        x_names=["D"],  # We'll vary embedding dimension
        x_vals=[128 * i for i in range(1, 9)],  # From 128 to 1024
        line_arg="provider",
        line_vals=["triton", "torch_vec", "torch_non_vec"],
        line_names=["Triton", "PyTorch Vectorized", "PyTorch Non-Vectorized"],
        styles=[("red", "-"), ("blue", "-"), ("green", "--")],
        ylabel="Execution Time (ms)",
        plot_name="apple-cce-performance-D",
        args={"N": 512, "V": 2048},  # Fixed dimensions
    ),
    triton.testing.Benchmark(
        x_names=["V"],  # We'll vary vocabulary size
        x_vals=[1000 * i for i in range(1, 11)],  # From 1000 to 10000
        line_arg="provider",
        line_vals=["triton", "torch_vec", "torch_non_vec"],
        line_names=["Triton", "PyTorch Vectorized", "PyTorch Non-Vectorized"],
        styles=[("red", "-"), ("blue", "-"), ("green", "--")],
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
    elif provider == 'torch_vec':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_indexed_essential_probs(E, C, I, use_vectorized=True), 
            quantiles=quantiles
        )
    elif provider == 'torch_non_vec':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_indexed_essential_probs(E, C, I, use_vectorized=False), 
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
        'Triton (ms)': [], 'PyTorch Vec (ms)': [], 'PyTorch Non-Vec (ms)': [],
        'Speedup vs Vec': [], 'Speedup vs Non-Vec': []
    }
    
    for shape in shapes:
        N, D, V = shape
        shape_str = f"N={N}, D={D}, V={V}"
        print(f"\nBenchmarking shape: {shape_str}")
        timings = benchmark_indexed_essential_probs(shape)
        
        # Extract timing values
        tri_ms = timings['triton'][0]
        torch_vec_ms = timings['torch_vec'][0]
        torch_non_vec_ms = timings['torch_non_vec'][0]
        
        # Calculate speedups
        speedup_vs_vec = torch_vec_ms / tri_ms
        speedup_vs_nonvec = torch_non_vec_ms / tri_ms
        
        # Store results
        results['Shape'].append(shape_str)
        results['N'].append(N)
        results['D'].append(D)
        results['V'].append(V)
        results['Triton (ms)'].append(f"{tri_ms:.2f}")
        results['PyTorch Vec (ms)'].append(f"{torch_vec_ms:.2f}")
        results['PyTorch Non-Vec (ms)'].append(f"{torch_non_vec_ms:.2f}")
        results['Speedup vs Vec'].append(f"{speedup_vs_vec:.2f}x")
        results['Speedup vs Non-Vec'].append(f"{speedup_vs_nonvec:.2f}x")
    
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
        print(f"  PyTorch Vec: {results['PyTorch Vec (ms)'][i]}ms (speedup: {results['Speedup vs Vec'][i]})")
        print(f"  PyTorch Non-Vec: {results['PyTorch Non-Vec (ms)'][i]}ms (speedup: {results['Speedup vs Non-Vec'][i]})")


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
            'torch_vec': [],
            'torch_non_vec': []
        }
        
        # Collect data for CSV
        csv_data = {
            x_name: [],
            'Triton (ms)': [],
            'PyTorch Vec (ms)': [],
            'PyTorch Non-Vec (ms)': [],
            'Speedup vs Vec': [],
            'Speedup vs Non-Vec': []
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
            
            # PyTorch vectorized
            torch_vec_ms, torch_vec_min_ms, torch_vec_max_ms = triton.testing.do_bench(
                lambda: torch_indexed_essential_probs(E, C, I, use_vectorized=True), 
                quantiles=quantiles
            )
            
            # PyTorch non-vectorized
            torch_non_vec_ms, torch_non_vec_min_ms, torch_non_vec_max_ms = triton.testing.do_bench(
                lambda: torch_indexed_essential_probs(E, C, I, use_vectorized=False), 
                quantiles=quantiles
            )
            
            # Store results
            data['triton'].append((x_val, tri_ms))
            data['torch_vec'].append((x_val, torch_vec_ms))
            data['torch_non_vec'].append((x_val, torch_non_vec_ms))
            
            # Store CSV data
            csv_data[x_name].append(x_val)
            csv_data['Triton (ms)'].append(f"{tri_ms * 1000:.2f}")
            csv_data['PyTorch Vec (ms)'].append(f"{torch_vec_ms * 1000:.2f}")
            csv_data['PyTorch Non-Vec (ms)'].append(f"{torch_non_vec_ms * 1000:.2f}")
            csv_data['Speedup vs Vec'].append(f"{torch_vec_ms/tri_ms:.2f}x")
            csv_data['Speedup vs Non-Vec'].append(f"{torch_non_vec_ms/tri_ms:.2f}x")
        
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
    torch_vec_times = [point[1] * 1000 for point in data['torch_vec']]
    torch_non_vec_times = [point[1] * 1000 for point in data['torch_non_vec']]
    
    # Create performance plot
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(x_vals, triton_times, 'ro-', label='Triton', linewidth=2)
    plt.plot(x_vals, torch_vec_times, 'bo-', label='PyTorch Vectorized', linewidth=2)
    plt.plot(x_vals, torch_non_vec_times, 'g--', label='PyTorch Non-Vectorized', linewidth=2, marker='s')
    
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
    vec_speedups = [vec/tri for vec, tri in zip(torch_vec_times, triton_times)]
    non_vec_speedups = [nonvec/tri for nonvec, tri in zip(torch_non_vec_times, triton_times)]
    
    # Plot speedups
    plt.plot(x_vals, vec_speedups, 'bo-', label='vs PyTorch Vectorized', linewidth=2)
    plt.plot(x_vals, non_vec_speedups, 'gs--', label='vs PyTorch Non-Vectorized', linewidth=2)
    
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
    
    # If PyTorch non-vectorized is much slower, create a separate plot without it
    if max(non_vec_speedups) > 5 * max(vec_speedups):
        plt.figure(figsize=(10, 6), dpi=150)
        plt.plot(x_vals, triton_times, 'ro-', label='Triton', linewidth=2)
        plt.plot(x_vals, torch_vec_times, 'bo-', label='PyTorch Vectorized', linewidth=2)
        
        plt.xlabel(x_name, fontsize=12)
        plt.ylabel('Execution Time (ms)', fontsize=12)
        plt.title(f'Apple CCE Performance - Vectorized Only (Varying {x_name})', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        vec_only_filename = f"apple_cce_vec_only_{x_name}_{timestamp}.png"
        plt.savefig(os.path.join('results', vec_only_filename), bbox_inches='tight')
        plt.close()
        print(f"Vectorized-only plot saved to results/{vec_only_filename}")


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