#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv) {

    int grids_nx[] = {250, 500, 1000};
    int grids_ny[] = {100, 200, 400};
    int points_range[] = {100, 10000, 1000000, 100000000};
    Maxiter = 10;

    printf("Experiment 01: Serial Mover Execution Time Scaling (Immediate)\n");
    printf("Grid, Particles, Immediate_Total_Time\n");

    for (int g = 0; g < 3; g++) {
        NX = grids_nx[g];
        NY = grids_ny[g];
        GRID_X = NX + 1;
        GRID_Y = NY + 1;
        dx = 1.0 / NX;
        dy = 1.0 / NY;

        double *mesh_value = NULL;
        posix_memalign((void**)&mesh_value, 64, GRID_X * GRID_Y * sizeof(double));

        for (int p = 0; p < 4; p++) {
            NUM_Points = points_range[p];

            Points *points = NULL;
            int alloc_status = posix_memalign((void**)&points, 64, (size_t)NUM_Points * sizeof(Points));

            if (alloc_status != 0 || !points) {
                printf("%dx%d, %d, OOM\n", NX, NY, NUM_Points);
                continue;
            }

            memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));
            memset(points, 0, (size_t)NUM_Points * sizeof(Points));
            initializepoints(points);

            double immediate_time = 0.0;
            for (int iter = 0; iter < Maxiter; iter++) {
                clock_t start = clock();
                interpolation(mesh_value, points);
                mover_serial_immediate(points, dx, dy);
                clock_t end = clock();
                immediate_time += (double)(end - start) / CLOCKS_PER_SEC;
            }

            printf("%dx%d, %d, %lf\n", NX, NY, NUM_Points, immediate_time);

            free(points);
        }
        free(mesh_value);
    }

    return 0;
}
