#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "init.h"
#include "utils.h"

// Global simulation parameters (extern in init.h)
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    // --- Read header on rank 0 and broadcast ---
    FILE *file = NULL;
    int params[4] = {0, 0, 0, 0};
    if (rank == 0) {
        file = fopen(argv[1], "rb");
        if (!file) {
            fprintf(stderr, "Error: cannot open %s\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fread(&params[0], sizeof(int), 1, file); // NX
        fread(&params[1], sizeof(int), 1, file); // NY
        fread(&params[2], sizeof(int), 1, file); // NUM_Points
        fread(&params[3], sizeof(int), 1, file); // Maxiter
    }
    MPI_Bcast(params, 4, MPI_INT, 0, MPI_COMM_WORLD);

    NX        = params[0];
    NY        = params[1];
    NUM_Points = params[2];
    Maxiter   = params[3];
    GRID_X    = NX + 1;
    GRID_Y    = NY + 1;
    dx        = 1.0 / NX;
    dy        = 1.0 / NY;
    int grid_size = GRID_X * GRID_Y;

    if (rank == 0) {
        printf("Grid: %dx%d  Particles: %d  Iterations: %d\n",
               NX, NY, NUM_Points, Maxiter);
        printf("MPI ranks: %d  OMP threads/rank: %d  Total cores: %d\n",
               size, omp_get_max_threads(), size * omp_get_max_threads());
    }

    // --- Compute per-rank particle range ---
    // Distribute NUM_Points particles evenly; first `rem` ranks get one extra.
    int base_n = NUM_Points / size;
    int rem    = NUM_Points % size;

    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs     = (int *)malloc(size * sizeof(int));
    {
        int off = 0;
        for (int r = 0; r < size; r++) {
            sendcounts[r] = base_n + (r < rem ? 1 : 0);
            displs[r]     = off;
            off          += sendcounts[r];
        }
    }
    int local_N = sendcounts[rank];

    // --- Rank 0 reads all particles, then scatter ---
    double *all_x = NULL, *all_y = NULL;
    if (rank == 0) {
        all_x = (double *)malloc(NUM_Points * sizeof(double));
        all_y = (double *)malloc(NUM_Points * sizeof(double));
        for (int i = 0; i < NUM_Points; i++) {
            fread(&all_x[i], sizeof(double), 1, file);
            fread(&all_y[i], sizeof(double), 1, file);
        }
        fclose(file);
    }

    double *local_x = (double *)malloc(local_N * sizeof(double));
    double *local_y = (double *)malloc(local_N * sizeof(double));

    MPI_Scatterv(all_x, sendcounts, displs, MPI_DOUBLE,
                 local_x, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(all_y, sendcounts, displs, MPI_DOUBLE,
                 local_y, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) { free(all_x); free(all_y); }

    // Build local Points array (all start active)
    Points *local_pts = (Points *)malloc(local_N * sizeof(Points));
    for (int i = 0; i < local_N; i++) {
        local_pts[i].x      = local_x[i];
        local_pts[i].y      = local_y[i];
        local_pts[i].active = 1;
    }
    free(local_x);
    free(local_y);

    // --- Allocate mesh buffers ---
    double *partial_mesh = (double *)malloc(grid_size * sizeof(double));
    double *global_mesh  = (double *)malloc(grid_size * sizeof(double));

    // --- Open CSV on rank 0 ---
    FILE *csv = NULL;
    if (rank == 0) {
        csv = fopen("results.csv", "a");
        if (csv) {
            fseek(csv, 0, SEEK_END);
            if (ftell(csv) == 0)
                fprintf(csv,
                        "NX,NY,num_particles,maxiter,mpi_ranks,omp_threads,"
                        "total_cores,iteration,interp_time_s,mover_time_s,"
                        "total_iter_time_s,active_particles\n");
        }
    }

    int omp_threads = omp_get_max_threads();
    int total_cores = size * omp_threads;

    // --- Main loop ---
    double wall_start    = MPI_Wtime();
    double sum_interp    = 0.0;
    double sum_mover     = 0.0;

    for (int iter = 0; iter < Maxiter; iter++) {
        double iter_start = MPI_Wtime();

        // 1. Zero partial mesh for this iteration
        memset(partial_mesh, 0, grid_size * sizeof(double));

        // 2. Forward interpolation: local particles → partial mesh
        double t0 = MPI_Wtime();
        interpolation_local(partial_mesh, local_pts, local_N);

        // 3. Allreduce partial meshes → global mesh (all ranks get full result)
        MPI_Allreduce(partial_mesh, global_mesh, grid_size,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double t_interp = MPI_Wtime() - t0;

        // 4. Normalize, mover, denormalize
        double t1 = MPI_Wtime();
        double fmin, fmax;
        normalize_mesh(global_mesh, grid_size, &fmin, &fmax);
        mover_local(global_mesh, local_pts, local_N);
        denormalize_mesh(global_mesh, grid_size, fmin, fmax);
        double t_mover = MPI_Wtime() - t1;

        double iter_time = MPI_Wtime() - iter_start;
        sum_interp += t_interp;
        sum_mover  += t_mover;

        // Count active particles across all ranks
        int local_active = 0;
        for (int i = 0; i < local_N; i++)
            if (local_pts[i].active) local_active++;
        int global_active = 0;
        MPI_Reduce(&local_active, &global_active, 1, MPI_INT,
                   MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Iter %2d | interp %7.4fs | mover %7.4fs | total %7.4fs | active %d\n",
                   iter + 1, t_interp, t_mover, iter_time, global_active);
            if (csv)
                fprintf(csv, "%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%d\n",
                        NX, NY, NUM_Points, Maxiter,
                        size, omp_threads, total_cores,
                        iter + 1, t_interp, t_mover, iter_time, global_active);
        }
    }

    double wall_total = MPI_Wtime() - wall_start;

    if (rank == 0) {
        printf("--\nTotal | interp %7.4fs | mover %7.4fs | wall %7.4fs\n",
               sum_interp, sum_mover, wall_total);
        save_mesh(global_mesh);
        if (csv) fclose(csv);
    }

    // --- Cleanup ---
    free(local_pts);
    free(partial_mesh);
    free(global_mesh);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}
