#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

// Cluster: Intel Xeon E5-2640 v3
// 2 sockets × 8 cores, 20 MB L3 per socket, 2 NUMA nodes
#define L3_PER_SOCKET    (20L * 1024 * 1024)
#define CORES_PER_SOCKET 8

// Persistent allocation: copies survive across interpolation() calls
// Eliminates calloc/free per iteration (page faults, mmap_lock contention)
static double **s_copies = NULL;
static int s_num_copies = 0;
static int s_grid_size = 0;
static int *s_worker_copy = NULL;

static void ensure_copies(int num_copies, int grid_size, int max_threads, int is_case1) {
    if (s_copies != NULL && s_num_copies == num_copies && s_grid_size == grid_size)
        return;

    // Free old allocations
    if (s_copies) {
        for (int c = 0; c < s_num_copies; c++) free(s_copies[c]);
        free(s_copies);
    }
    if (s_worker_copy) { free(s_worker_copy); s_worker_copy = NULL; }

    s_copies = (double **) malloc(num_copies * sizeof(double *));
    s_num_copies = num_copies;
    s_grid_size = grid_size;

    if (is_case1) {
        // CASE 1: Allocate in parallel for NUMA first-touch
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid < num_copies)
                s_copies[tid] = (double *) calloc(grid_size, sizeof(double));
        }
    } else {
        // CASE 2: Strided worker assignment for NUMA balance
        s_worker_copy = (int *) malloc(max_threads * sizeof(int));
        for (int t = 0; t < max_threads; t++)
            s_worker_copy[t] = -1;
        for (int c = 0; c < num_copies; c++)
            s_worker_copy[c * max_threads / num_copies] = c;

        // Workers allocate their own copy (first-touch NUMA placement)
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int my_copy = s_worker_copy[tid];
            if (my_copy >= 0)
                s_copies[my_copy] = (double *) calloc(grid_size, sizeof(double));
        }
    }
}

void interpolation(double *mesh_value, Points *points) {
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;
    const double cell_area = dx * dy;
    const int grid_size = GRID_X * GRID_Y;
    const long mesh_bytes = (long)grid_size * sizeof(double);

    const int max_threads = omp_get_max_threads();

    // Per-socket L3 budget → scale with sockets
    int copies_per_socket = (int)(L3_PER_SOCKET / mesh_bytes);
    if (copies_per_socket < 1) copies_per_socket = 1;
    if (copies_per_socket > CORES_PER_SOCKET) copies_per_socket = CORES_PER_SOCKET;

    int num_sockets = (max_threads + CORES_PER_SOCKET - 1) / CORES_PER_SOCKET;
    int num_copies = copies_per_socket * num_sockets;
    if (num_copies > max_threads) num_copies = max_threads;

    const int is_case1 = (num_copies >= max_threads);

    ensure_copies(num_copies, grid_size, max_threads, is_case1);

    if (is_case1) {
        // CASE 1: Pure privatization — every thread has its own copy
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double * __restrict__ my_mesh = s_copies[tid];

            memset(my_mesh, 0, grid_size * sizeof(double));

            // No barrier needed — each thread writes only to its own copy

            #pragma omp for schedule(static) nowait
            for (int k = 0; k < NUM_Points; k++) {
                double sx = points[k].x * inv_dx;
                double sy = points[k].y * inv_dy;
                int col = (int)sx;
                int row = (int)sy;
                if (col >= NX) col = NX - 1;
                if (row >= NY) row = NY - 1;
                double fx = sx - col;
                double fy = sy - row;
                double fx_ca  = fx * cell_area;
                double ifx_ca = cell_area - fx_ca;
                int base = row * GRID_X + col;
                my_mesh[base]              += ifx_ca * (1.0 - fy);
                my_mesh[base + 1]          += fx_ca  * (1.0 - fy);
                my_mesh[base + GRID_X]     += ifx_ca * fy;
                my_mesh[base + GRID_X + 1] += fx_ca  * fy;
            }

            #pragma omp barrier

            #pragma omp for schedule(static)
            for (int i = 0; i < grid_size; i++) {
                for (int c = 0; c < num_copies; c++)
                    mesh_value[i] += s_copies[c][i];
            }
        }
    } else {
        // CASE 2: Limited privatization — strided workers across NUMA nodes
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int my_copy = s_worker_copy[tid];

            if (my_copy >= 0) {
                double * __restrict__ my_mesh = s_copies[my_copy];

                memset(my_mesh, 0, grid_size * sizeof(double));

                // Manual partition among workers
                int ppw = NUM_Points / num_copies;
                int start = my_copy * ppw;
                int end = (my_copy == num_copies - 1) ? NUM_Points : start + ppw;

                for (int k = start; k < end; k++) {
                    double sx = points[k].x * inv_dx;
                    double sy = points[k].y * inv_dy;
                    int col = (int)sx;
                    int row = (int)sy;
                    if (col >= NX) col = NX - 1;
                    if (row >= NY) row = NY - 1;
                    double fx = sx - col;
                    double fy = sy - row;
                    double fx_ca  = fx * cell_area;
                    double ifx_ca = cell_area - fx_ca;
                    int base = row * GRID_X + col;
                    my_mesh[base]              += ifx_ca * (1.0 - fy);
                    my_mesh[base + 1]          += fx_ca  * (1.0 - fy);
                    my_mesh[base + GRID_X]     += ifx_ca * fy;
                    my_mesh[base + GRID_X + 1] += fx_ca  * fy;
                }
            }

            #pragma omp barrier

            // ALL threads participate in reduction
            #pragma omp for schedule(static)
            for (int i = 0; i < grid_size; i++) {
                for (int c = 0; c < num_copies; c++)
                    mesh_value[i] += s_copies[c][i];
            }
        }
    }
}

// Write mesh to file
void save_mesh(double *mesh_value) {
    FILE *fd = fopen("Mesh.out", "w");
    if (!fd) { printf("Error creating Mesh.out\n"); exit(1); }
    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++)
            fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        fprintf(fd, "\n");
    }
    fclose(fd);
}
