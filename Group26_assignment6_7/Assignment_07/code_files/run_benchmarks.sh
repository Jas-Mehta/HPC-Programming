#!/bin/bash
# Run all benchmarks for Cluster
# Generates each input file, runs all thread counts, then deletes it before next config.
# This keeps disk usage to one input file at a time.

set -e

make clean
make input_maker
make

THREADS="1 2 4 8 16"
OUTFILE="results_cluster.csv"

echo "config,threads,interp_time,norm_time,mover_time,denorm_time,total_time,voids" > "$OUTFILE"

run_config() {
    local cfg=$1
    local NX=$2
    local NY=$3
    local NPTS=$4
    local INPUT="input_${cfg}.bin"

    echo ">>> Generating Config ($cfg): Nx=${NX}, Ny=${NY}, Points=${NPTS}, Maxiter=10"
    echo -e "${NX} ${NY}\n${NPTS}\n10" | ./input_file_maker
    mv input.bin "$INPUT"
    echo ">>> Generated: $(du -h $INPUT | cut -f1) - $INPUT"

    for t in $THREADS; do
        echo "=== Config ($cfg) | Threads=$t ==="
        export OMP_NUM_THREADS=$t

        OUTPUT=$(./interpolation "$INPUT" 2>&1)
        echo "$OUTPUT"

        INTERP=$(echo "$OUTPUT" | grep "Total Interpolation Time" | awk '{print $5}')
        NORM=$(echo "$OUTPUT"   | grep "Total Normalization Time" | awk '{print $5}')
        MOVER=$(echo "$OUTPUT"  | grep "Total Mover Time"         | awk '{print $5}')
        DENORM=$(echo "$OUTPUT" | grep "Total Denormalization Time" | awk '{print $5}')
        TOTAL=$(echo "$OUTPUT"  | grep "Total Algorithm Time"     | awk '{print $5}')
        VOIDS=$(echo "$OUTPUT"  | grep "Total Number of Voids"    | awk '{print $6}')

        echo "${cfg},${t},${INTERP},${NORM},${MOVER},${DENORM},${TOTAL},${VOIDS}" >> "$OUTFILE"
        echo ""
    done

    rm -f "$INPUT"
    echo ">>> Deleted $INPUT to free disk space"
    echo ""
}

# Config (a): Nx=250,  Ny=100, Points=900000
run_config "a" 250 100 900000

# Config (b): Nx=250,  Ny=100, Points=5000000
run_config "b" 250 100 5000000

# Config (c): Nx=500,  Ny=200, Points=3600000
run_config "c" 500 200 3600000

# Config (d): Nx=500,  Ny=200, Points=20000000
run_config "d" 500 200 20000000

# Config (e): Nx=1000, Ny=400, Points=14000000
run_config "e" 1000 400 14000000

echo "============================================"
echo "All benchmarks complete! Results: $OUTFILE"
echo "============================================"
cat "$OUTFILE"
