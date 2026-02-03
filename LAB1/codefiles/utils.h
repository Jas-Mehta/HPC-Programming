#ifndef UTILS_H
#define UTILS_H

// Standard STREAM Kernels
void kernel_copy(double *x, double *y, int Np);
void kernel_scale(double *x, double *y, double scalar, int Np);
void kernel_add(double *x, double *y, double *s, int Np);
void kernel_triad(double *x, double *y, double *s, double scalar, int Np);

// Helper to clear cache (optional but good for strict testing)
void dummy(int x);

// Analysis Kernels
void kernel_triad_memory(double *x, double *y, double *s, double scalar, int Np);
void kernel_triad_compute(double *x, double *y, double *s, double scalar, int Np);

#endif