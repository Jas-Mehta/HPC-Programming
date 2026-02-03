#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "init.h"
#include "utils.h"

#define CLK CLOCK_MONOTONIC

// Helper to time a specific kernel
double time_kernel(int kernel_type, int Np, int RUNS, double *x, double *y, double *v, double *S, double scalar) {
    struct timespec start, end;
    clock_gettime(CLK, &start);

    for (int r = 0; r < RUNS; r++) {
        if (kernel_type == 0) kernel_copy(S, x, Np);
        else if (kernel_type == 1) kernel_scale(S, x, scalar, Np);
        else if (kernel_type == 2) kernel_add(x, y, S, Np);
        else if (kernel_type == 3) kernel_triad(x, y, S, scalar, Np);
        else if (kernel_type == 4) kernel_triad_memory(x, y, S, scalar, Np);
        else if (kernel_type == 5) kernel_triad_compute(x, y, S, scalar, Np);
    }

    clock_gettime(CLK, &end);
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
}

int main() {
    int minProbSize = 1 << 8;   // 2^8 per PDF
    int maxProbSize = 1 << 26;  // 2^26 Safe for RAM
    
    double *x, *y, *v, *S;
    double scalar = 3.0;

    printf("Kernel, ProblemSize, RUNS, TotalOps, Time\n");

    for (int Np = minProbSize; Np <= maxProbSize; Np *= 2) {
        
        // Initialize
        init_vectors(Np, &x, &y, &v, &S);
        
        // Dynamic RUNS
        int RUNS = (maxProbSize / Np) + 1;
        if (RUNS < 10) RUNS = 10; 

        // Update list of kernels
        const char* names[] = {"Copy", "Scale", "Add", "Triad", "TriadMem", "TriadComp"};

        // Loop 0 to 5 now
        for (int k = 0; k < 6; k++) {
            double time = time_kernel(k, Np, RUNS, x, y, v, S, scalar);
            printf("%s, %d, %d, %lld, %.9lf\n", names[k], Np, RUNS, (long long)Np * RUNS, time);
        }

        free(x); free(y); free(v); free(S);
    }

    return 0;
}