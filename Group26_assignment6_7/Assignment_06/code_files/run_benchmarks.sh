#!/bin/bash
# ============================================================
# Benchmark Script for Assignment 6: Parallel Interpolation
# Cluster: Intel Xeon E5-2640 v3 (2x8 cores, 2 NUMA nodes)
# Generates input files, runs with 1,2,4,8,16 threads
# Outputs CSV results for easy plotting
# ============================================================

set -e

# Compile everything
echo "=== Compiling ==="
make clean
make
echo ""

# Validate correctness first
echo "=== Validating correctness with Test_input.bin ==="
for T in 1 2 4 8 16; do
    OMP_NUM_THREADS=$T OMP_PROC_BIND=close OMP_PLACES=cores ./parallel.out Test_input.bin > /dev/null 2>&1
    if diff <(sed 's/ *$//' Mesh.out) <(sed 's/ *$//' Test_Mesh.out) > /dev/null 2>&1; then
        echo "  Threads=$T: PASS"
    else
        echo "  Threads=$T: FAIL — output mismatch!"
        exit 1
    fi
done
echo ""

# Configuration array: "label NX NY Points Maxiter"
CONFIGS=(
    "a 250 100 900000 10"
    "b 250 100 5000000 10"
    "c 500 200 3600000 10"
    "d 500 200 20000000 10"
    "e 1000 400 14000000 10"
)

THREADS=(1 2 4 8 16)

# Output CSV file
RESULTS="benchmark_results.csv"
echo "config,NX,NY,points,threads,time_seconds" > $RESULTS

for CONFIG in "${CONFIGS[@]}"; do
    read -r LABEL NX NY POINTS MAXITER <<< "$CONFIG"

    echo "=== Config $LABEL: NX=$NX, NY=$NY, Points=$POINTS, Maxiter=$MAXITER ==="

    # Generate input file by piping values to input_maker
    echo -e "${NX} ${NY}\n${POINTS}\n${MAXITER}" | ./input_maker.out
    echo ""

    for T in "${THREADS[@]}"; do
        echo -n "  Threads=$T: "

        # Run and capture the time output
        OUTPUT=$(OMP_NUM_THREADS=$T OMP_PROC_BIND=close OMP_PLACES=cores ./parallel.out input.bin 2>&1)
        TIME=$(echo "$OUTPUT" | grep "Total interpolation time" | awk '{print $(NF-1)}')

        echo "${TIME}s"
        echo "$LABEL,$NX,$NY,$POINTS,$T,$TIME" >> $RESULTS
    done

    echo ""
done

echo "=== Results saved to $RESULTS ==="
echo ""
cat $RESULTS

echo ""
echo "=== Done ==="
