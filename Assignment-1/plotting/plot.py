import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("results/benchmark_results.csv")

# 1. Bandwidth Plot (Standard STREAM)
plt.figure(figsize=(10, 6))
stream_kernels = ["Copy", "Scale", "Add", "Triad"]
for kernel in stream_kernels:
    subset = df[df["Kernel"] == kernel]
    plt.plot(np.log2(subset["ProblemSize"]), subset["AvgBandwidth_GBs"], marker='o', label=kernel)

plt.xlabel("log2(Working Set Size)")
plt.ylabel("Bandwidth (GB/s)")
plt.title("LAB1: Memory Bandwidth vs Size")
plt.legend()
plt.grid(True)
plt.savefig("results/bandwidth_plot.png")
print("Saved bandwidth_plot.png")

# 2. Performance Plot (MFLOPs)
plt.figure(figsize=(10, 6))
comp_kernels = ["Scale", "Add", "Triad"]
for kernel in comp_kernels:
    subset = df[df["Kernel"] == kernel]
    plt.plot(np.log2(subset["ProblemSize"]), subset["AvgMFLOPS"], marker='s', label=kernel)

plt.xlabel("log2(Working Set Size)")
plt.ylabel("Performance (MFLOPs)")
plt.title("LAB1: Compute Performance vs Size")
plt.legend()
plt.grid(True)
plt.savefig("results/performance_plot.png")
print("Saved performance_plot.png")

# 3. Compute vs Memory Split
plt.figure(figsize=(10, 6))
split_kernels = ["Triad", "TriadMem", "TriadComp"]
for kernel in split_kernels:
    subset = df[df["Kernel"] == kernel]
    # Calculate Time per Element (seconds) * 1e9 = nanoseconds
    tpe = (subset["AvgTime"] / subset["TotalOps"]) * 1e9
    plt.plot(np.log2(subset["ProblemSize"]), tpe, marker='x', label=kernel)

plt.xlabel("log2(Working Set Size)")
plt.ylabel("Time per Element (ns)")
plt.title("LAB1: Compute vs Memory Time Analysis")
plt.legend()
plt.grid(True)
plt.savefig("results/compute_split_plot.png")
print("Saved compute_split_plot.png")