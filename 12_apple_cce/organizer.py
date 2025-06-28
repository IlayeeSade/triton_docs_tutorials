import os
import glob
import pandas as pd
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
import re


def organize_results(source_dir='results', output_dir='organized_results'):
    os.makedirs(output_dir, exist_ok=True)
    
    csv_dir = os.path.join(output_dir, 'csv_data')
    plot_dir = os.path.join(output_dir, 'plots')
    table_dir = os.path.join(output_dir, 'tables')
    
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    
    all_files = glob.glob(os.path.join(source_dir, '*.*'))
    csv_files = [f for f in all_files if f.endswith('.csv')]
    png_files = [f for f in all_files if f.endswith('.png')]
    
    print(f"Found {len(csv_files)} CSV files in {source_dir}: {csv_files}")
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        if 'iep' in filename or 'lsemm' in filename:
            if 'basic_benchmarks' in filename:
                shutil.copy(csv_file, os.path.join(csv_dir, filename))
                print(f"Copied basic benchmark: {csv_file} to {csv_dir}")
            else:
                match = re.search(r'(iep|lsemm)-performance-(\w+)_\w+_(\d+)', filename)
                if match:
                    algo, param, timestamp = match.group(1), match.group(2), match.group(3)
                    param_dir = os.path.join(csv_dir, f'{algo}_detailed_{param}')
                    os.makedirs(param_dir, exist_ok=True)
                    dest = os.path.join(param_dir, filename)
                    shutil.copy(csv_file, dest)
                    print(f"Copied detailed benchmark: {csv_file} to {dest}")
                else:
                    print(f"Skipped unrecognized CSV: {csv_file}")
    
    for png_file in png_files:
        filename = os.path.basename(png_file)
        if 'performance' in filename and 'speedup' not in filename:
            plot_type_dir = os.path.join(plot_dir, 'performance')
            os.makedirs(plot_type_dir, exist_ok=True)
            shutil.copy(png_file, os.path.join(plot_type_dir, filename))
        elif 'speedup' in filename:
            plot_type_dir = os.path.join(plot_dir, 'speedup')
            os.makedirs(plot_type_dir, exist_ok=True)
            shutil.copy(png_file, os.path.join(plot_dir, 'speedup', filename))
        else:
            shutil.copy(png_file, os.path.join(table_dir, filename))
    
    print(f"Organized {len(csv_files)} CSV files and {len(png_files)} PNG files from {source_dir}")
    generate_summary_report(csv_dir, output_dir, plot_dir)


def generate_summary_report(csv_dir, output_dir, plot_dir):
    report_file = os.path.join(output_dir, 'summary_report.html')
    csv_files = glob.glob(os.path.join(csv_dir, '**/*.csv'), recursive=True)
    print(f"Found {len(csv_files)} total CSV files in {csv_dir}: {csv_files}")
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Apple CCE Benchmark Summary</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            h2 { color: #555; margin-top: 30px; }
            h3 { color: #666; margin-top: 20px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .summary { margin-top: 30px; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }
            .plot-description { margin-bottom: 15px; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #4CAF50; }
            .plot-container { margin: 20px 0; text-align: center; }
            .plot-container img { max-width: 100%; height: auto; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>Apple CCE Benchmark Summary</h1>
        <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        <div class="summary">
            <h2>Overview</h2>
            <p>This report summarizes benchmark results for two algorithms:</p>
            <ul>
                <li><strong>Indexed Essential Probs (IEP)</strong>: Computes specific indexed probabilities</li>
                <li><strong>Log-Sum-Exp Matrix Multiplication (LSEMM/LSEMMO)</strong>: Computes log-sum-exp of matrix products</li>
            </ul>
            <p>Implementations compared:</p>
            <ul>
                <li><strong>Triton (IEP)</strong>: Optimized Triton implementation for IEP</li>
                <li><strong>Triton LSEMM</strong>: First Triton implementation for LSEMM</li>
                <li><strong>Triton LSEMMO</strong>: Alternative Triton implementation for LSEMM</li>
                <li><strong>PyTorch</strong>: Reference implementation using PyTorch operations</li>
            </ul>
        </div>
    """
    
    for algo in ['iep', 'lsemm']:
        algo_files = [f for f in csv_files if algo in os.path.basename(f)]
        print(f"Processing {algo} with {len(algo_files)} files: {algo_files}")
        
        html_content += f"<h2>{'Indexed Essential Probs (IEP)' if algo == 'iep' else 'LSEMM/LSEMMO'} Results</h2>"
        
        basic_files = [f for f in algo_files if 'basic_benchmarks' in os.path.basename(f)]
        if basic_files:
            html_content += "<h3>Basic Benchmarks</h3>"
            for csv_file in basic_files:
                df = pd.read_csv(csv_file)
                rel_path = os.path.relpath(csv_file, csv_dir)
                html_content += f"""
                <div class="plot-description">
                    <p>Results from {rel_path}</p>
                    <p>Basic benchmarks compare performance across different shape configurations.</p>
                    <ul>
                        <li><strong>Shape</strong>: Dimensions (N=batch size, D=embedding dimension, V=vocabulary size)</li>
                        {'<li><strong>Triton (ms)</strong>: Execution time for Triton IEP</li>' if algo == 'iep' else 
                         '<li><strong>Triton LSEMM (ms)</strong>: Execution time for Triton LSEMM</li><li><strong>Triton LSEMMO (ms)</strong>: Execution time for Triton LSEMMO</li>'}
                        <li><strong>PyTorch (ms)</strong>: Execution time for PyTorch</li>
                        {'<li><strong>Speedup</strong>: Triton vs PyTorch speedup</li>' if algo == 'iep' else 
                         '<li><strong>Speedup LSEMM</strong>: Triton LSEMM vs PyTorch</li><li><strong>Speedup LSEMMO</strong>: Triton LSEMMO vs PyTorch</li>'}
                    </ul>
                </div>
                """
                html_content += df.to_html(index=False)
    
        detailed_files = [f for f in algo_files if 'basic_benchmarks' not in os.path.basename(f)]
        if detailed_files:
            html_content += "<h3>Detailed Benchmarks</h3>"
            for csv_file in detailed_files:
                df = pd.read_csv(csv_file)
                rel_path = os.path.relpath(csv_file, csv_dir)
                match = re.search(r'(iep|lsemm)-performance-(\w+)_\w+_(\d+)', os.path.basename(csv_file))
                if match:
                    param, timestamp = match.group(2), match.group(3)
                    param_desc = {'N': 'batch size', 'D': 'embedding dimension', 'V': 'vocabulary size'}.get(param, param)
                    html_content += f"""
                    <div class="plot-description">
                        <p>Results from {rel_path}</p>
                        <p>Detailed benchmarks varying {param_desc}:</p>
                        <ul>
                            <li><strong>{param}</strong>: Varying {param_desc}</li>
                            {'<li><strong>Triton (ms)</strong>: Execution time for Triton IEP</li>' if algo == 'iep' else 
                             '<li><strong>Triton LSEMM (ms)</strong>: Execution time for Triton LSEMM</li><li><strong>Triton LSEMMO (ms)</strong>: Execution time for Triton LSEMMO</li>'}
                            <li><strong>PyTorch (ms)</strong>: Execution time for PyTorch</li>
                            {'<li><strong>Speedup</strong>: Triton vs PyTorch speedup</li>' if algo == 'iep' else 
                             '<li><strong>Speedup LSEMM</strong>: Triton LSEMM vs PyTorch</li><li><strong>Speedup LSEMMO</strong>: Triton LSEMMO vs PyTorch</li>'}
                        </ul>
                    </div>
                    """
                    html_content += df.to_html(index=False)
                    performance_plot = glob.glob(os.path.join(plot_dir, 'performance', f'{algo}-performance-{param}_{param}_{timestamp}.png'))
                    if performance_plot:
                        rel_path = os.path.relpath(performance_plot[0], output_dir)
                        html_content += f"""
                        <div class="plot-container">
                            <h4>Performance Plot for {param}</h4>
                            <img src="{rel_path}" alt="Performance plot for {param}">
                        </div>
                        """
    
    html_content += """
    <h2>Combined Performance Plots</h2>
    <div class="plot-description">
        <p>Combined plots aggregate results from multiple benchmark runs.</p>
    </div>
    """
    
    combined_plots_dir = os.path.join(plot_dir, 'combined')
    os.makedirs(combined_plots_dir, exist_ok=True)
    combined_plots = glob.glob(os.path.join(combined_plots_dir, 'combined_*.png'))
    print(f"Looking for combined plots in {combined_plots_dir}")
    print(f"Found {len(combined_plots)} combined plots: {combined_plots}")
    
    if combined_plots:
        for plot in sorted(combined_plots):
            rel_path = os.path.relpath(plot, output_dir)
            plot_name = os.path.basename(plot)
            match = re.search(r'combined_(iep|lsemm)_(\w+)_performance', plot_name)
            if match:
                algo, param = match.group(1), match.group(2)
                html_content += f"""
                <div class="plot-container">
                    <h3>Combined {algo.upper()} Plot for {param}</h3>
                    <img src="{rel_path}" alt="Combined {algo} performance plot for {param}">
                </div>
                """
            else:
                print(f"Skipped unrecognized combined plot: {plot}")
    else:
        html_content += "<p>No combined plots found. Check if plots are generated correctly.</p>"
    
    html_content += "</body></html>"
    
    with open(report_file, 'w') as f:
        f.write(html_content)
    print(f"Summary report generated at {report_file}")


def create_combined_plots(source_dir='organized_results/csv_data', output_dir='organized_results'):
    plot_dir = os.path.join(output_dir, 'plots', 'combined')
    os.makedirs(plot_dir, exist_ok=True)
    
    for algo in ['iep', 'lsemm']:
        csv_files = glob.glob(os.path.join(source_dir, f'{algo}_detailed_*/*.csv'))
        print(f"Found {len(csv_files)} {algo} detailed CSV files in {source_dir}: {csv_files}")
        
        param_files = {}
        for csv_file in csv_files:
            match = re.search(rf'{algo}-performance-(\w+)_\w+_(\d+)', os.path.basename(csv_file))
            if match:
                param = match.group(1)
                param_files.setdefault(param, []).append(csv_file)
        
        print(f"Parameters found for {algo}: {list(param_files.keys())}")
        for param, files in param_files.items():
            print(f"Processing {algo} {param} with {len(files)} files: {files}")
            if len(files) > 1:
                create_combined_plot_for_param(algo, param, files, plot_dir)
            else:
                print(f"Skipping combined plot for {algo} {param}: only {len(files)} run(s) found (need >1)")


def create_combined_plot_for_param(algo, param, csv_files, plot_dir):
    plt.figure(figsize=(10, 6), dpi=150)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for i, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            print(f"Columns in {csv_file}: {df.columns.tolist()}")
            timestamp = os.path.basename(csv_file).split('_')[-1].replace('.csv', '')
            x_vals = df[param].values  # Correctly uses 'V', 'N', 'D'
            if algo == 'iep':
                if 'Triton (ms)' not in df.columns:
                    print(f"Warning: 'Triton (ms)' not found in {csv_file}")
                    continue
                triton_times = pd.to_numeric(df['Triton (ms)'], errors='coerce').values
                plt.plot(x_vals, triton_times, marker=markers[i % len(markers)], color=colors[i % len(colors)],
                         linestyle='-', linewidth=2, label=f'IEP Run {timestamp}')
            else:
                if 'Triton LSEMM (ms)' not in df.columns or 'Triton LSEMMO (ms)' not in df.columns:
                    print(f"Warning: 'Triton LSEMM (ms)' or 'Triton LSEMMO (ms)' not found in {csv_file}")
                    continue
                lsemm_times = pd.to_numeric(df['Triton LSEMM (ms)'], errors='coerce').values
                lsemmo_times = pd.to_numeric(df['Triton LSEMMO (ms)'], errors='coerce').values
                plt.plot(x_vals, lsemm_times, marker=markers[i % len(markers)], color=colors[i % len(colors)],
                         linestyle='-', linewidth=2, label=f'LSEMM Run {timestamp}')
                plt.plot(x_vals, lsemmo_times, marker=markers[(i+1) % len(markers)], color=colors[(i+1) % len(colors)],
                         linestyle='--', linewidth=2, label=f'LSEMMO Run {timestamp}')
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    plt.xlabel(param, fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title(f'Combined {algo.upper()} Performance (Varying {param})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    combined_filename = f"combined_{algo}_{param}_performance.png"
    full_path = os.path.join(plot_dir, combined_filename)
    plt.savefig(full_path, bbox_inches='tight')
    plt.close()
    print(f"Combined plot for {algo} {param} saved to {full_path}")


if __name__ == "__main__":
    organize_results()
    create_combined_plots()