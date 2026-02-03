import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
try:
    df = pd.read_csv('benchmark_results.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'benchmark_results.csv' not found. Please ensure the file is in the working directory.")
    exit()

# Ensure consistent sorting by ProblemSize
df = df.sort_values(by='ProblemSize')

# Define a standard formatter for the plots
def setup_plot(title, ylabel, xlabel='Working Data Size (Elements)'):
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.xscale('log', base=2)
    plt.legend()
    plt.tight_layout()

# ==========================================
# 1. Combined Bandwidth Plot (Task 2)
# ==========================================
plt.figure(figsize=(10, 6))

kernels = ['Copy', 'Scale', 'Add', 'Triad']
colors = ['blue', 'orange', 'green', 'red']

for kernel, color in zip(kernels, colors):
    subset = df[df['Kernel'] == kernel]
    if not subset.empty:
        plt.plot(subset['ProblemSize'], subset['AvgBandwidth_GBs'], 
                 marker='o', label=kernel, color=color)

setup_plot('Memory Bandwidth vs Data Size', 'Bandwidth (GB/s)')
plt.savefig('bandwidth_combined.png')
plt.show()

# ==========================================
# 2. Combined Performance Plot (Task 3)
# ==========================================
plt.figure(figsize=(10, 6))

# Usually Copy is excluded from FLOPs charts as it has 0 FLOPs
perf_kernels = ['Scale', 'Add', 'Triad']
perf_colors = ['orange', 'green', 'red']

for kernel, color in zip(perf_kernels, perf_colors):
    subset = df[df['Kernel'] == kernel]
    if not subset.empty:
        plt.plot(subset['ProblemSize'], subset['AvgMFLOPS'], 
                 marker='s', label=kernel, color=color)

setup_plot('Computational Performance vs Data Size', 'Performance (MFLOPS)')
plt.savefig('performance_combined.png')
plt.show()

# ==========================================
# 3. Triad Compute vs Memory Split (Task 4)
# ==========================================
# This analyzes the time breakdown: Triad (Total) vs TriadComp (Compute only) vs TriadMem (Memory only)
plt.figure(figsize=(10, 6))

split_kernels = ['Triad', 'TriadComp', 'TriadMem']
split_labels = ['Total Triad Time', 'Compute Only', 'Memory Access Only']
split_colors = ['black', 'red', 'blue']

for kernel, label, color in zip(split_kernels, split_labels, split_colors):
    subset = df[df['Kernel'] == kernel]
    if not subset.empty:
        plt.plot(subset['ProblemSize'], subset['AvgTime'], 
                 marker='x', label=label, color=color, linestyle='--')

setup_plot('Triad Execution Time Analysis (Split)', 'Execution Time (seconds)')
plt.savefig('triad_split_analysis.png')
plt.show()

# ==========================================
# 4. Individual Plots (User Request)
# ==========================================

# List of individual kernels to plot
individual_kernels = ['Triad', 'Scale', 'Add']

for kernel in individual_kernels:
    subset = df[df['Kernel'] == kernel]
    
    if not subset.empty:
        plt.figure(figsize=(8, 5))
        plt.plot(subset['ProblemSize'], subset['AvgBandwidth_GBs'], 
                 marker='o', label=f'{kernel} Bandwidth', color='purple')
        
        setup_plot(f'{kernel} Kernel Bandwidth', 'Bandwidth (GB/s)')
        
        # Save each plot with a unique name
        filename = f'{kernel.lower()}_bandwidth_plot.png'
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.show()