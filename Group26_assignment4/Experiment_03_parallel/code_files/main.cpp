#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY, NUM_Points, Maxiter;
double dx, dy;

int main() {
    NX = 1000; NY = 400; Maxiter = 10; NUM_Points = 14000000;
    GRID_X = NX + 1; GRID_Y = NY + 1;
    dx = 1.0 / NX; dy = 1.0 / NY;
    omp_set_num_threads(4);

    double *mesh_value = NULL;
    Points *points = NULL;
    posix_memalign((void**)&mesh_value, 64, GRID_X * GRID_Y * sizeof(double));
    posix_memalign((void**)&points, 64, (size_t)NUM_Points * sizeof(Points));

    #pragma omp parallel for schedule(static)
    for (int p = 0; p < NUM_Points; p++) {
        points[p].x = 0.0;
        points[p].y = 0.0;
    }
    memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));
    initializepoints(points);

    printf("Iter, Interp_Time, Mover_Parallel_Time, Total_Time, Speedup\n");

    for (int iter = 0; iter < Maxiter; iter++) {
        double w_start_interp = omp_get_wtime();
        interpolation(mesh_value, points);
        double w_end_interp = omp_get_wtime();

        double w_start_move_s = omp_get_wtime();
        mover_serial(points, dx, dy);
        double w_end_move_s = omp_get_wtime();

        double w_start_move_p = omp_get_wtime();
        mover_parallel(points, dx, dy);
        double w_end_move_p = omp_get_wtime();

        double w_interp = w_end_interp - w_start_interp;
        double w_move_s = w_end_move_s - w_start_move_s;
        double w_move_p = w_end_move_p - w_start_move_p;

        printf("%d, %lf, %lf, %lf, %lf\n", iter + 1, w_interp, w_move_p, w_interp + w_move_p, w_move_s / w_move_p);
    }
    free(mesh_value); free(points);
    return 0;
}