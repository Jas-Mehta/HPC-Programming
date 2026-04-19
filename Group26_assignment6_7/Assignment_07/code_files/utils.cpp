#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

double min_val, max_val;

// Persistent thread-private mesh storage (avoids repeated malloc/free)
static double **thread_meshes = NULL;
static int alloc_nthreads = 0;
static int alloc_grid_size = 0;

static void ensure_thread_meshes(int num_copies, int grid_size) {
    if (thread_meshes != NULL && alloc_nthreads >= num_copies && alloc_grid_size >= grid_size)
        return;
    if (thread_meshes) {
        for (int t = 0; t < alloc_nthreads; t++) free(thread_meshes[t]);
        free(thread_meshes);
    }
    thread_meshes = (double **)malloc(num_copies * sizeof(double *));
    for (int t = 0; t < num_copies; t++) {
        thread_meshes[t] = (double *)malloc(grid_size * sizeof(double));
    }
    alloc_nthreads = num_copies;
    alloc_grid_size = grid_size;
}

// Scatter interpolation: point -> mesh
// Strategy: grouped mesh privatization
//   - Compute how many private mesh copies fit in L3 cache (~14MB budget)
//   - If enough for 1-per-thread: pure privatization (no sync in hot loop)
//   - If fewer: group threads onto shared mesh copies with atomics (low contention)
//   - Merge uses cache-friendly sequential passes
void interpolation(double *mesh_value, Points *points) {
    int grid_size = GRID_X * GRID_Y;
    memset(mesh_value, 0, grid_size * sizeof(double));

    int nthreads;
    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
    }

    // How many mesh copies fit in ~14MB of L3 cache?
    int max_copies = (int)(14LL * 1024 * 1024 / ((long long)grid_size * sizeof(double)));
    if (max_copies < 1) max_copies = 1;
    int num_copies = (nthreads <= max_copies) ? nthreads : max_copies;

    ensure_thread_meshes(num_copies, grid_size);

    // Zero mesh copies in parallel
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if (tid < num_copies) {
            memset(thread_meshes[tid], 0, grid_size * sizeof(double));
        }
    }

    if (num_copies == nthreads) {
        // === Full privatization: one mesh per thread, no atomics ===
        #pragma omp parallel
        {
            double *local_mesh = thread_meshes[omp_get_thread_num()];

            #pragma omp for schedule(static)
            for (int p = 0; p < NUM_Points; p++) {
                if (points[p].is_void) continue;

                double x = points[p].x;
                double y = points[p].y;

                int i = (int)(x / dx);
                int j = (int)(y / dy);
                if (i >= NX) i = NX - 1;
                if (j >= NY) j = NY - 1;

                double lx = x - i * dx;
                double ly = y - j * dy;

                double w_ij   = (dx - lx) * (dy - ly);
                double w_i1j  = ly * (dx - lx);
                double w_ij1  = lx * (dy - ly);
                double w_i1j1 = lx * ly;

                local_mesh[j       * GRID_X + i    ] += w_ij;
                local_mesh[j       * GRID_X + i + 1] += w_i1j;
                local_mesh[(j + 1) * GRID_X + i    ] += w_ij1;
                local_mesh[(j + 1) * GRID_X + i + 1] += w_i1j1;
            }
        }
    } else {
        // === Grouped privatization: multiple threads share mesh copies ===
        // Threads are spread across copies (tid % num_copies) so partners
        // process distant particle ranges -> minimal contention on atomics
        #pragma omp parallel
        {
            int mesh_id = omp_get_thread_num() % num_copies;
            double *local_mesh = thread_meshes[mesh_id];

            #pragma omp for schedule(static)
            for (int p = 0; p < NUM_Points; p++) {
                if (points[p].is_void) continue;

                double x = points[p].x;
                double y = points[p].y;

                int i = (int)(x / dx);
                int j = (int)(y / dy);
                if (i >= NX) i = NX - 1;
                if (j >= NY) j = NY - 1;

                double lx = x - i * dx;
                double ly = y - j * dy;

                double w_ij   = (dx - lx) * (dy - ly);
                double w_i1j  = ly * (dx - lx);
                double w_ij1  = lx * (dy - ly);
                double w_i1j1 = lx * ly;

                #pragma omp atomic
                local_mesh[j       * GRID_X + i    ] += w_ij;
                #pragma omp atomic
                local_mesh[j       * GRID_X + i + 1] += w_i1j;
                #pragma omp atomic
                local_mesh[(j + 1) * GRID_X + i    ] += w_ij1;
                #pragma omp atomic
                local_mesh[(j + 1) * GRID_X + i + 1] += w_i1j1;
            }
        }
    }

    // Cache-friendly merge: one mesh copy at a time
    for (int t = 0; t < num_copies; t++) {
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < grid_size; k++) {
            mesh_value[k] += thread_meshes[t][k];
        }
    }
}

// Normalize mesh values to [-1, 1]
void normalization(double *mesh_value) {
    int grid_size = GRID_X * GRID_Y;

    double local_min = mesh_value[0];
    double local_max = mesh_value[0];

    #pragma omp parallel for reduction(min:local_min) reduction(max:local_max) schedule(static)
    for (int k = 0; k < grid_size; k++) {
        if (mesh_value[k] < local_min) local_min = mesh_value[k];
        if (mesh_value[k] > local_max) local_max = mesh_value[k];
    }

    min_val = local_min;
    max_val = local_max;

    double range = max_val - min_val;
    if (range == 0.0) return;

    double inv_range = 2.0 / range;

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < grid_size; k++) {
        mesh_value[k] = (mesh_value[k] - min_val) * inv_range - 1.0;
    }
}

// Gather reverse-interpolation: mesh -> point, then update positions
// Read-only mesh access -> no race conditions, embarrassingly parallel
void mover(double *mesh_value, Points *points) {

    #pragma omp parallel for schedule(static)
    for (int p = 0; p < NUM_Points; p++) {
        if (points[p].is_void) continue;

        double x = points[p].x;
        double y = points[p].y;

        int i = (int)(x / dx);
        int j = (int)(y / dy);
        if (i >= NX) i = NX - 1;
        if (j >= NY) j = NY - 1;

        double lx = x - i * dx;
        double ly = y - j * dy;

        double w_ij   = (dx - lx) * (dy - ly);
        double w_i1j  = ly * (dx - lx);
        double w_ij1  = lx * (dy - ly);
        double w_i1j1 = lx * ly;

        double Fi = w_ij   * mesh_value[j       * GRID_X + i    ]
                  + w_i1j  * mesh_value[j       * GRID_X + i + 1]
                  + w_ij1  * mesh_value[(j + 1) * GRID_X + i    ]
                  + w_i1j1 * mesh_value[(j + 1) * GRID_X + i + 1];

        double new_x = x + Fi * dx;
        double new_y = y + Fi * dy;

        if (new_x < 0.0 || new_x > 1.0 || new_y < 0.0 || new_y > 1.0) {
            points[p].is_void = true;
        } else {
            points[p].x = new_x;
            points[p].y = new_y;
        }
    }
}

// Denormalize mesh values back to original range
void denormalization(double *mesh_value) {
    int grid_size = GRID_X * GRID_Y;
    double range = max_val - min_val;
    if (range == 0.0) return;

    double half_range = range / 2.0;

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < grid_size; k++) {
        mesh_value[k] = (mesh_value[k] + 1.0) * half_range + min_val;
    }
}

// Count particles that went beyond the domain
long long int void_count(Points *points) {
    long long int voids = 0;

    #pragma omp parallel for reduction(+:voids) schedule(static)
    for (int i = 0; i < NUM_Points; i++) {
        voids += (int)points[i].is_void;
    }
    return voids;
}

// Write mesh to file
void save_mesh(double *mesh_value) {
    FILE *fd = fopen("Mesh.out", "w");
    if (!fd) {
        printf("Error creating Mesh.out\n");
        exit(1);
    }

    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        }
        fprintf(fd, "\n");
    }

    fclose(fd);
}
