#include <math.h>
#include "utils.h"

// 1. Copy: x = y
void kernel_copy(double *x, double *y, int Np) {
    for (int i = 0; i < Np; i++) {
        x[i] = y[i];
    }
}

// 2. Scale: x = scalar * y
void kernel_scale(double *x, double *y, double scalar, int Np) {
    for (int i = 0; i < Np; i++) {
        x[i] = scalar * y[i];
    }
}

// 3. Add: s = x + y
void kernel_add(double *x, double *y, double *s, int Np) {
    for (int i = 0; i < Np; i++) {
        s[i] = x[i] + y[i];
    }
}

// 4. Triad: s = x + scalar * y
void kernel_triad(double *x, double *y, double *s, double scalar, int Np) {
    for (int i = 0; i < Np; i++) {
        s[i] = x[i] + scalar * y[i];
    }
}

// 5. Memory Only: Same loads/stores as Triad, but minimal math
// 5. Memory Only: Force reads of X and Y, and write to S, with MINIMAL math
void kernel_triad_memory(double *x, double *y, double *s, double scalar, int Np) {
    // We use a tiny scalar (effectively 0) to minimize floating point complexity
    // but the compiler can't verify it's zero at compile time easily if passed in.
    // Ideally, just doing an Add is the best proxy for memory traffic.
    
    for (int i = 0; i < Np; i++) {
        // Simple Add (A+B) has the same memory traffic as Triad (A+s*B)
        // But removes the Multiplication instruction.
        s[i] = x[i] + y[i]; 
    }
}

// 6. Compute Only: Same Math as Triad, but minimized memory traffic 
// (We cheat by doing the math many times on local registers)
// 6. Compute Only: Perform the Triad Math, but WITHOUT touching main memory arrays
// Optimized with unrolling to allow ILP (Instruction Level Parallelism)
void kernel_triad_compute(double *x, double *y, double *s, double scalar, int Np) {
    // Load reliable initial values
    double a1 = x[0], b1 = y[0];
    double a2 = x[1], b2 = y[1];
    double a3 = x[2], b3 = y[2];
    double a4 = x[3], b4 = y[3];
    
    // Accumulators for parallel execution
    double r1 = 0, r2 = 0, r3 = 0, r4 = 0;

    // Unroll loop 4x to break strict serial dependencies
    for (int i = 0; i < Np; i += 4) {
        r1 += a1 + scalar * b1;
        r2 += a2 + scalar * b2;
        r3 += a3 + scalar * b3;
        r4 += a4 + scalar * b4;
    }
    
    // Prevent optimization
    s[0] = r1 + r2 + r3 + r4;
}

void dummy(int x) {
    volatile int sink = x;
}