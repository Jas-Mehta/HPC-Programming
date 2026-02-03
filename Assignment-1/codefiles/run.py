import os
import subprocess
import csv

EXECUTABLE = "./main"
OUTPUT_CSV = "results/benchmark_results.csv"
NUM_RUNS = 5 

os.makedirs("results", exist_ok=True)

# Dictionary to store aggregated results: results[kernel][size] = [times...]
data = {}

print(f"Running benchmark {NUM_RUNS} times...")

# Compile first!
print("Compiling...")
compile_cmd = ["g++", "main.cpp", "init.cpp", "utils.cpp", "-o", "main", "-O3"]
try:
    subprocess.run(compile_cmd, check=True)
    print("Compilation successful.")
except subprocess.CalledProcessError:
    print("Error: Compilation failed. Make sure you have g++ installed.")
    exit(1)
except FileNotFoundError:
    print("Error: g++ not found. Please install a C++ compiler (g++).")
    exit(1)

for i in range(NUM_RUNS):
    print(f"Iteration {i+1}...")
    proc = subprocess.run([EXECUTABLE], stdout=subprocess.PIPE, text=True)
    
    lines = proc.stdout.strip().split("\n")
    for line in lines[1:]: # Skip header
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5: continue
        
        kernel = parts[0]
        size = int(parts[1])
        runs = int(parts[2])
        time = float(parts[4])
        
        if kernel not in data: data[kernel] = {}
        if size not in data[kernel]: data[kernel][size] = []
        
        data[kernel][size].append((runs, time))

# Write averaged results
with open(OUTPUT_CSV, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Kernel", "ProblemSize", "AvgBandwidth_GBs", "AvgMFLOPS", "AvgTime", "TotalOps"])
    
    for kernel in data:
        for size in sorted(data[kernel].keys()):
            entries = data[kernel][size]
            avg_time = sum(e[1] for e in entries) / len(entries)
            runs = entries[0][0]
            
            # Bandwidth Calculation (GB/s)
            bytes_per_op = 0
            if kernel in ["Copy", "Scale"]: bytes_per_op = 16
            elif kernel in ["Add", "Triad", "TriadMem"]: bytes_per_op = 24
            
            total_bytes = size * runs * bytes_per_op
            bw = (total_bytes / avg_time) / 1e9

            # MFLOPs Calculation
            flops_per_op = 0
            if kernel in ["Scale", "Add"]: flops_per_op = 1
            elif kernel == "Triad": flops_per_op = 2
            elif kernel == "TriadComp": flops_per_op = 2 # Matches standard Triad
            
            total_flops = size * runs * flops_per_op
            mflops = (total_flops / avg_time) / 1e6
            
            writer.writerow([kernel, size, f"{bw:.4f}", f"{mflops:.4f}", f"{avg_time:.6f}", size*runs])

print(f"Results saved to {OUTPUT_CSV}")