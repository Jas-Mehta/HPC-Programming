#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <omp.h>
#include "utils.h"

// -----------------------------------------------------------------------
// Persistent private mesh copies (one per OMP thread, NUMA first-touch).
// Allocated once per (num_threads, grid_size) combination; reused across
// iterations to avoid calloc/page-fault overhead every call.
// -----------------------------------------------------------------------
static double **s_priv  = NULL;
static int      s_nthrd = 0;
static int      s_gsz   = 0;

static void ensure_priv(int nthrd, int gsz) {
    if (s_priv && s_nthrd == nthrd && s_gsz == gsz)
        return;

    // Free old allocation
    if (s_priv) {
        for (int t = 0; t < s_nthrd; t++) free(s_priv[t]);
        free(s_priv);
        s_priv = NULL;
    }

    s_priv  = (double **)malloc(nthrd * sizeof(double *));
    s_nthrd = nthrd;
    s_gsz   = gsz;

    // Allocate each thread's private copy from within a parallel region so
    // the OS first-touches the pages on the NUMA node where the thread runs.
    #pragma omp parallel num_threads(nthrd)
    {
        int tid     = omp_get_thread_num();
        s_priv[tid] = (double *)calloc(gsz, sizeof(double));
    }
}

// -----------------------------------------------------------------------
// Forward interpolation: local particles → partial_mesh.
// partial_mesh must be pre-zeroed by the caller.
// Uses per-thread privatisation to avoid race conditions; parallel
// reduction at the end merges all private copies.
// -----------------------------------------------------------------------
void interpolation_local(double *partial_mesh, Points *points, int n_pts) {
    const double inv_dx   = 1.0 / dx;
    const double inv_dy   = 1.0 / dy;
    const double cell_area = dx * dy;
    const int    gsz       = GRID_X * GRID_Y;
    const int    nthrd     = omp_get_max_threads();

    if (nthrd == 1) {
        // --- Serial fast-path: accumulate directly, no privatisation needed ---
        for (int k = 0; k < n_pts; k++) {
            if (!points[k].active) continue;
            double sx  = points[k].x * inv_dx;
            double sy  = points[k].y * inv_dy;
            int    col = (int)sx;  if (col >= NX) col = NX - 1;
            int    row = (int)sy;  if (row >= NY) row = NY - 1;
            double fx  = sx - col;
            double fy  = sy - row;
            double fxA = fx * cell_area;
            double ifA = cell_area - fxA;
            int    b   = row * GRID_X + col;
            partial_mesh[b]              += ifA * (1.0 - fy);
            partial_mesh[b + 1]          += fxA * (1.0 - fy);
            partial_mesh[b + GRID_X]     += ifA * fy;
            partial_mesh[b + GRID_X + 1] += fxA * fy;
        }
        return;
    }

    // --- Parallel path: per-thread private copy + reduction ---
    ensure_priv(nthrd, gsz);

    #pragma omp parallel num_threads(nthrd)
    {
        int     tid      = omp_get_thread_num();
        double *my_mesh  = s_priv[tid];

        // Zero this thread's private copy
        memset(my_mesh, 0, gsz * sizeof(double));

        // Scatter: each thread accumulates into its own copy — no races
        #pragma omp for schedule(static) nowait
        for (int k = 0; k < n_pts; k++) {
            if (!points[k].active) continue;
            double sx  = points[k].x * inv_dx;
            double sy  = points[k].y * inv_dy;
            int    col = (int)sx;  if (col >= NX) col = NX - 1;
            int    row = (int)sy;  if (row >= NY) row = NY - 1;
            double fx  = sx - col;
            double fy  = sy - row;
            double fxA = fx * cell_area;
            double ifA = cell_area - fxA;
            int    b   = row * GRID_X + col;
            my_mesh[b]              += ifA * (1.0 - fy);
            my_mesh[b + 1]          += fxA * (1.0 - fy);
            my_mesh[b + GRID_X]     += ifA * fy;
            my_mesh[b + GRID_X + 1] += fxA * fy;
        }

        // Barrier: all threads must finish writing before reduction reads
        #pragma omp barrier

        // Reduction: sum all private copies into partial_mesh.
        // The omp for distributes grid indices, so each mesh cell is
        // written by exactly one thread — no atomics needed.
        #pragma omp for schedule(static)
        for (int i = 0; i < gsz; i++) {
            double sum = 0.0;
            for (int t = 0; t < nthrd; t++)
                sum += s_priv[t][i];
            partial_mesh[i] += sum;
        }
    }
}

// -----------------------------------------------------------------------
// Normalise mesh values to [-1, 1].
// Returns the original fmin/fmax so the caller can denormalise later.
// -----------------------------------------------------------------------
void normalize_mesh(double *mesh, int gsz, double *out_fmin, double *out_fmax) {
    double fmin =  DBL_MAX;
    double fmax = -DBL_MAX;

    #pragma omp parallel for reduction(min:fmin) reduction(max:fmax) schedule(static)
    for (int i = 0; i < gsz; i++) {
        if (mesh[i] < fmin) fmin = mesh[i];
        if (mesh[i] > fmax) fmax = mesh[i];
    }

    *out_fmin = fmin;
    *out_fmax = fmax;

    double range = fmax - fmin;
    if (range < 1e-15) {
        // All values identical — set everything to 0 (mid of [-1,1])
        memset(mesh, 0, gsz * sizeof(double));
        return;
    }
    double inv_range = 2.0 / range;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < gsz; i++)
        mesh[i] = (mesh[i] - fmin) * inv_range - 1.0;
}

// -----------------------------------------------------------------------
// Restore mesh from normalised [-1,1] back to original scale.
// -----------------------------------------------------------------------
void denormalize_mesh(double *mesh, int gsz, double fmin, double fmax) {
    double range = fmax - fmin;
    if (range < 1e-15) {
        // Was flat — restore to fmin
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < gsz; i++)
            mesh[i] = fmin;
        return;
    }
    double half_range = range * 0.5;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < gsz; i++)
        mesh[i] = (mesh[i] + 1.0) * half_range + fmin;
}

// -----------------------------------------------------------------------
// Mover: reverse interpolation (grid → particle) + position update.
//
// The normalised mesh is in [-1,1]. For each active particle we compute
// a field value F_i using standard bilinear interpolation (weights sum to
// 1), then displace the particle by (F_i*dx, F_i*dy). Particles that
// leave [0,1]^2 are deactivated.
//
// This is embarrassingly parallel — each particle index is independent.
// -----------------------------------------------------------------------
void mover_local(double *mesh, Points *points, int n_pts) {
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n_pts; k++) {
        if (!points[k].active) continue;

        double sx  = points[k].x * inv_dx;
        double sy  = points[k].y * inv_dy;
        int    col = (int)sx;  if (col >= NX) col = NX - 1;
        int    row = (int)sy;  if (row >= NY) row = NY - 1;

        // Bilinear weights (normalised, sum to 1)
        double fx = sx - col;   // fractional x in [0,1]
        double fy = sy - row;   // fractional y in [0,1]

        int b = row * GRID_X + col;
        double Fi = (1.0 - fx) * (1.0 - fy) * mesh[b]
                  +        fx  * (1.0 - fy) * mesh[b + 1]
                  + (1.0 - fx) *        fy  * mesh[b + GRID_X]
                  +        fx  *        fy  * mesh[b + GRID_X + 1];

        double xn = points[k].x + Fi * dx;
        double yn = points[k].y + Fi * dy;

        if (xn < 0.0 || xn > 1.0 || yn < 0.0 || yn > 1.0) {
            points[k].active = 0;
        } else {
            points[k].x = xn;
            points[k].y = yn;
        }
    }
}

// -----------------------------------------------------------------------
// Write final mesh to Mesh.out (row-major, space-separated).
// -----------------------------------------------------------------------
void save_mesh(double *mesh) {
    FILE *fd = fopen("Mesh.out", "w");
    if (!fd) { fprintf(stderr, "Error: cannot create Mesh.out\n"); return; }
    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++)
            fprintf(fd, "%.6f ", mesh[i * GRID_X + j]);
        fprintf(fd, "\n");
    }
    fclose(fd);
}
