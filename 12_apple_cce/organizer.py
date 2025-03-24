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
        elif 'vec_only' in filename:
            plot_type_dir = os.path.join(plot_dir, 'vectorized_only')
            os.makedirs(plot_type_dir, exist_ok=True)
            shutil.copy(png_file, os.path.join(plot_type_dir, filename))
        elif 'detailed' in filename and not ('performance' in filename or 'speedup' in filename):
            # This is likely a summary table
            shutil.copy(png_file, os.path.join(table_dir, filename))
        else:
            # Other PNG files
            shutil.copy(png_file, os.path.join(plot_dir, filename))
    
    # Generate summary report
    generate_summary_report(csv_dir, output_dir)
    
    print(f"Results organized in {output_dir}")


def generate_summary_report(csv_dir, output_dir):
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
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .summary { margin-top: 30px; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Apple CCE Benchmark Summary</h1>
        <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        <div class="summary">
            <h2>Overview</h2>
            <p>This report summarizes the benchmark results for the Apple CCE implementation.</p>
        </div>
    """
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            rel_path = os.path.relpath(csv_file, csv_dir)
            
            html_content += f"<h2>Results from {rel_path}</h2>"
            
            # Add table
            html_content += df.to_html(index=False)
            
            # Add summary statistics if available
            if 'Speedup vs Vec' in df.columns:
                # Convert to numeric, handling potential string formatting
                speedup_col = df['Speedup vs Vec'].str.replace('x', '').astype(float)
                avg_speedup = speedup_col.mean()
                max_speedup = speedup_col.max()
                
                html_content += f"""
                <div class="summary">
                    <p><strong>Average Speedup vs Vectorized PyTorch:</strong> {avg_speedup:.2f}x</p>
                    <p><strong>Maximum Speedup vs Vectorized PyTorch:</strong> {max_speedup:.2f}x</p>
                </div>
                """
        except Exception as e:
            html_content += f"<p>Error processing {csv_file}: {str(e)}</p>"
    
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
    plt.figure(figsize=(12, 8), dpi=150)
    
    # Use different markers and colors for different runs
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for i, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            timestamp = os.path.basename(csv_file).split('_')[-1].replace('.csv', '')
            
            # Extract data
            x_vals = df[param].values
            triton_times = df['Triton (ms)'].str.replace('ms', '').astype(float)
            
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
    
    # Save plot
    os.makedirs(os.path.join(output_dir, 'combined_plots'), exist_ok=True)
    combined_filename = f"combined_{param}_performance.png"
    plt.savefig(os.path.join(output_dir, 'combined_plots', combined_filename), bbox_inches='tight')
    plt.close()
    
    print(f"Combined plot for {param} saved to {os.path.join(output_dir, 'combined_plots', combined_filename)}")


if __name__ == "__main__":
    # Organize results
    organize_results()
    
    # Create combined plots from multiple runs
    create_combined_plots()
