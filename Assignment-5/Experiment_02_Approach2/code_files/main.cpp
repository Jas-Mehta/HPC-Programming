#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv) {

    int grids_nx[] = {250, 500, 1000};
    int grids_ny[] = {100, 200, 400};
    int threads[] = {2, 4, 8, 16};

    Maxiter = 10;
    NUM_Points = 14000000;

    printf("Experiment 02: Parallel Scalability (Immediate)\n");
    printf("Grid, Threads, Ass4_Serial, Ass4_Parallel, Immediate_Serial, Immediate_Parallel\n");

    for (int g = 0; g < 3; g++) {
        NX = grids_nx[g];
        NY = grids_ny[g];
        GRID_X = NX + 1;
        GRID_Y = NY + 1;
        dx = 1.0 / NX;
        dy = 1.0 / NY;

        double *mesh_value = NULL;
        Points *points = NULL;
        posix_memalign((void**)&mesh_value, 64, GRID_X * GRID_Y * sizeof(double));
        posix_memalign((void**)&points, 64, (size_t)NUM_Points * sizeof(Points));

        // Calculate Baseline Serial Times
        memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

        initializepoints(points);
        double serial_ass4 = 0.0;
        for (int i = 0; i < Maxiter; i++) {
            double t1 = omp_get_wtime();
            mover_serial_ass4(points, dx, dy);
            serial_ass4 += omp_get_wtime() - t1;
        }

        initializepoints(points);
        double serial_imm = 0.0;
        for (int i = 0; i < Maxiter; i++) {
            double t1 = omp_get_wtime();
            mover_serial_immediate(points, dx, dy);
            serial_imm += omp_get_wtime() - t1;
        }

        // Loop Through Thread Counts
        for (int t = 0; t < 4; t++) {
            omp_set_num_threads(threads[t]);

            initializepoints(points);
            double par_ass4 = 0.0;
            for (int i = 0; i < Maxiter; i++) {
                double t1 = omp_get_wtime();
                mover_parallel_ass4(points, dx, dy);
                par_ass4 += omp_get_wtime() - t1;
            }

            initializepoints(points);
            double par_imm = 0.0;
            for (int i = 0; i < Maxiter; i++) {
                double t1 = omp_get_wtime();
                mover_parallel_immediate(points, dx, dy);
                par_imm += omp_get_wtime() - t1;
            }

            printf("%dx%d, %d, %lf, %lf, %lf, %lf\n",
                   NX, NY, threads[t],
                   serial_ass4, par_ass4,
                   serial_imm, par_imm);
        }

        free(points);
        free(mesh_value);
    }

    return 0;
}
