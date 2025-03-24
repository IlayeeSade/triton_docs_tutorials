import os
import glob
import pandas as pd
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
import re


def organize_results(source_dir='results', output_dir='organized_results'):
    """
    Organize benchmark results into a structured directory format.
    
    Args:
        source_dir: Directory containing the raw benchmark results
        output_dir: Directory where organized results will be stored
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for different result types
    csv_dir = os.path.join(output_dir, 'csv_data')
    plot_dir = os.path.join(output_dir, 'plots')
    table_dir = os.path.join(output_dir, 'tables')
    
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    
    # Find all result files
    all_files = glob.glob(os.path.join(source_dir, '*.*'))
    
    # Categorize files
    csv_files = [f for f in all_files if f.endswith('.csv')]
    png_files = [f for f in all_files if f.endswith('.png')]
    
    # Process CSV files
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        # Determine if it's a basic or detailed benchmark
        if 'detailed' in filename:
            # Extract the parameter being varied
            match = re.search(r'detailed_(\w+)_\d+', filename)
            if match:
                param = match.group(1)
                param_dir = os.path.join(csv_dir, f'detailed_{param}')
                os.makedirs(param_dir, exist_ok=True)
                shutil.copy(csv_file, os.path.join(param_dir, filename))
            else:
                shutil.copy(csv_file, os.path.join(csv_dir, filename))
        else:
            # Basic benchmark results
            shutil.copy(csv_file, os.path.join(csv_dir, filename))
    
    # Process PNG files
    for png_file in png_files:
        filename = os.path.basename(png_file)
        # Determine the type of plot
        if 'performance' in filename:
            plot_type_dir = os.path.join(plot_dir, 'performance')
            os.makedirs(plot_type_dir, exist_ok=True)
            shutil.copy(png_file, os.path.join(plot_type_dir, filename))
        elif 'speedup' in filename:
            plot_type_dir = os.path.join(plot_dir, 'speedup')
            os.makedirs(plot_type_dir, exist_ok=True)
            shutil.copy(png_file, os.path.join(plot_type_dir, filename))
        else:
            # Other PNG files
            shutil.copy(png_file, os.path.join(plot_dir, filename))
    
    # Generate summary report
    generate_summary_report(csv_dir, output_dir, plot_dir)
    
    print(f"Results organized in {output_dir}")


def generate_summary_report(csv_dir, output_dir, plot_dir):
    """Generate a summary report of all benchmark results"""
    report_file = os.path.join(output_dir, 'summary_report.html')
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(csv_dir, '**/*.csv'), recursive=True)
    
    # Start HTML content
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
            <p>This report summarizes the benchmark results for the Apple CCE implementation.</p>
            <p>The benchmarks compare two implementations:</p>
            <ul>
                <li><strong>Triton</strong>: Our optimized implementation using the Triton framework</li>
                <li><strong>PyTorch</strong>: A vectorized implementation using PyTorch operations</li>
            </ul>
        </div>
    """
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            rel_path = os.path.relpath(csv_file, csv_dir)
            filename = os.path.basename(csv_file)
            
            html_content += f"<h2>Results from {rel_path}</h2>"
            
            # Add description based on file type
            if 'basic_benchmarks' in filename:
                html_content += """
                <div class="plot-description">
                    <h3>Basic Benchmarks Table</h3>
                    <p>This table shows performance comparisons across different shape configurations where all dimensions (N, D, V) are scaled together.</p>
                    <ul>
                        <li><strong>Shape</strong>: The dimensions used for the benchmark (N=batch size, D=embedding dimension, V=vocabulary size)</li>
                        <li><strong>Triton (ms)</strong>: Execution time for the Triton implementation in milliseconds</li>
                        <li><strong>PyTorch (ms)</strong>: Execution time for the PyTorch implementation in milliseconds</li>
                        <li><strong>Speedup</strong>: How many times faster Triton is compared to PyTorch</li>
                    </ul>
                </div>
                """
            elif 'detailed' in filename:
                # Extract the parameter being varied
                match = re.search(r'detailed_(\w+)', filename)
                if match:
                    param = match.group(1)
                    param_description = {
                        'N': 'batch size',
                        'D': 'embedding dimension',
                        'V': 'vocabulary size'
                    }.get(param, param)
                    
                    html_content += f"""
                    <div class="plot-description">
                        <h3>Detailed Benchmark Table (Varying {param})</h3>
                        <p>This table shows performance metrics when varying the {param_description} while keeping other dimensions fixed.</p>
                        <p>The x-axis represents different values of {param}, and the y-axis (in the corresponding plots) shows execution time in milliseconds.</p>
                        <p>Each implementation (Triton, PyTorch) is represented as a different data series.</p>
                        <ul>
                            <li><strong>{param}</strong>: The varying {param_description} value</li>
                            <li><strong>Triton (ms)</strong>: Execution time for the Triton implementation in milliseconds</li>
                            <li><strong>PyTorch (ms)</strong>: Execution time for the PyTorch implementation in milliseconds</li>
                            <li><strong>Speedup</strong>: How many times faster Triton is compared to PyTorch</li>
                        </ul>
                    </div>
                    """
            
            # Add table
            html_content += df.to_html(index=False)
            
            # Add summary statistics if available
            if 'Speedup' in df.columns:
                # Convert to numeric, handling potential string formatting
                speedup_col = pd.to_numeric(df['Speedup'], errors='coerce')
                avg_speedup = speedup_col.mean()
                max_speedup = speedup_col.max()
                
                html_content += f"""
                <div class="summary">
                    <p><strong>Average Speedup vs PyTorch:</strong> {avg_speedup:.2f}x</p>
                    <p><strong>Maximum Speedup vs PyTorch:</strong> {max_speedup:.2f}x</p>
                </div>
                """
                
                # Add plot descriptions and actual plots based on file type
                if 'detailed' in filename:
                    match = re.search(r'detailed_(\w+)_(\d+)', filename)
                    if match:
                        param = match.group(1)
                        timestamp = match.group(2)
                        
                        # Find corresponding plot files
                        performance_plot = glob.glob(os.path.join(plot_dir, 'performance', f'*performance_{param}_{timestamp}.png'))
                        speedup_plot = glob.glob(os.path.join(plot_dir, 'speedup', f'*speedup_{param}_{timestamp}.png'))
                        
                        html_content += f"""
                        <h3>Associated Plots</h3>
                        <div class="plot-description">
                            <h4>Performance Plot</h4>
                            <p>The performance plot shows execution time (ms) on the y-axis versus {param} values on the x-axis.</p>
                            <p>Lower values indicate better performance. The plot includes two lines:</p>
                            <ul>
                                <li><strong>Triton</strong>: Our optimized Triton implementation</li>
                                <li><strong>PyTorch</strong>: The PyTorch implementation</li>
                            </ul>
                        </div>
                        """
                        
                        # Insert performance plot if found
                        if performance_plot:
                            rel_path = os.path.relpath(performance_plot[0], output_dir)
                            html_content += f"""
                            <div class="plot-container">
                                <img src="{rel_path}" alt="Performance plot for {param}">
                            </div>
                            """
                        
                        html_content += f"""
                        <div class="plot-description">
                            <h4>Speedup Plot</h4>
                            <p>The speedup plot shows the relative performance gain of Triton compared to PyTorch implementation.</p>
                            <p>The y-axis shows speedup factor (higher is better), and the x-axis shows {param} values.</p>
                            <ul>
                                <li><strong>Speedup vs PyTorch</strong>: How many times faster Triton is compared to PyTorch</li>
                                <li><strong>Baseline (Equal Performance)</strong>: Reference line at y=1.0</li>
                            </ul>
                        </div>
                        """
                        
                        # Insert speedup plot if found
                        if speedup_plot:
                            rel_path = os.path.relpath(speedup_plot[0], output_dir)
                            html_content += f"""
                            <div class="plot-container">
                                <img src="{rel_path}" alt="Speedup plot for {param}">
                            </div>
                            """
        except Exception as e:
            html_content += f"<p>Error processing {csv_file}: {str(e)}</p>"
    
    # Add combined plots section
    html_content += """
    <h2>Combined Performance Plots</h2>
    <div class="plot-description">
        <p>Combined plots aggregate results from multiple benchmark runs to show performance consistency and trends.</p>
        <p>Each line represents a different benchmark run, with the x-axis showing the varying parameter and the y-axis showing execution time in milliseconds.</p>
        <p>Different colors and markers are used to distinguish between runs.</p>
    </div>
    """
    
    # Find and insert combined plots
    combined_plots_dir = os.path.join(output_dir, 'combined_plots')
    if os.path.exists(combined_plots_dir):
        combined_plots = glob.glob(os.path.join(combined_plots_dir, '*.png'))
        for plot in combined_plots:
            rel_path = os.path.relpath(plot, output_dir)
            plot_name = os.path.basename(plot)
            param = plot_name.split('_')[1] if '_' in plot_name else 'parameter'
            
            html_content += f"""
            <div class="plot-container">
                <h3>Combined Plot for {param}</h3>
                <img src="{rel_path}" alt="Combined performance plot for {param}">
            </div>
            """
    else:
        html_content += "<p>No combined plots found.</p>"
    
    # Close HTML
    html_content += """
    </body>
    </html>
    """
    
    # Write to file
    with open(report_file, 'w') as f:
        f.write(html_content)
    
    print(f"Summary report generated at {report_file}")


def create_combined_plots(source_dir='results', output_dir='organized_results'):
    """Create combined plots from multiple benchmark runs"""
    # Find all detailed CSV files
    csv_files = glob.glob(os.path.join(source_dir, 'apple_cce_detailed_*.csv'))
    
    # Group by parameter
    param_files = {}
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        match = re.search(r'detailed_(\w+)_\d+', filename)
        if match:
            param = match.group(1)
            if param not in param_files:
                param_files[param] = []
            param_files[param].append(csv_file)
    
    # Create combined plots for each parameter
    for param, files in param_files.items():
        if len(files) > 1:
            create_combined_plot_for_param(param, files, output_dir)


def create_combined_plot_for_param(param, csv_files, output_dir):
    """Create a combined plot for a specific parameter from multiple CSV files"""
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Use different markers and colors for different runs
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for i, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            timestamp = os.path.basename(csv_file).split('_')[-1].replace('.csv', '')
            
            # Extract data
            x_vals = df[param].values
            triton_times = df['Triton (ms)'].values
            
            # Plot with different style for each run
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            plt.plot(x_vals, triton_times, marker=marker, color=color, 
                     linestyle='-', linewidth=2, label=f'Run {timestamp}')
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
    
    # Add labels and title
    plt.xlabel(param, fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title(f'Combined Apple CCE Performance (Varying {param})', fontsize=14)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Create directory if it doesn't exist
    combined_plots_dir = os.path.join(output_dir, 'combined_plots')
    os.makedirs(combined_plots_dir, exist_ok=True)
    
    # Save plot
    combined_filename = f"combined_{param}_performance.png"
    plt.savefig(os.path.join(combined_plots_dir, combined_filename), bbox_inches='tight')
    plt.close()
    
    print(f"Combined plot for {param} saved to {os.path.join(combined_plots_dir, combined_filename)}")


if __name__ == "__main__":
    # Organize results
    organize_results()
    
    # Create combined plots from multiple runs
    create_combined_plots()
