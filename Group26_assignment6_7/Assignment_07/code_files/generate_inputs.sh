#!/bin/bash
# Generate all 5 input configurations for benchmarking

set -e

make input_maker

echo "Generating Config (a): Nx=250, Ny=100, Points=900000, Maxiter=10"
echo -e "250 100\n900000\n10" | ./input_file_maker
mv input.bin input_a.bin

echo "Generating Config (b): Nx=250, Ny=100, Points=5000000, Maxiter=10"
echo -e "250 100\n5000000\n10" | ./input_file_maker
mv input.bin input_b.bin

echo "Generating Config (c): Nx=500, Ny=200, Points=3600000, Maxiter=10"
echo -e "500 200\n3600000\n10" | ./input_file_maker
mv input.bin input_c.bin

echo "Generating Config (d): Nx=500, Ny=200, Points=20000000, Maxiter=10"
echo -e "500 200\n20000000\n10" | ./input_file_maker
mv input.bin input_d.bin

echo "Generating Config (e): Nx=1000, Ny=400, Points=14000000, Maxiter=10"
echo -e "1000 400\n14000000\n10" | ./input_file_maker
mv input.bin input_e.bin

echo ""
echo "All input files generated successfully!"
ls -lh input_*.bin
