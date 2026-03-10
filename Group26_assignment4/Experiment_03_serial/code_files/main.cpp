#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY, NUM_Points, Maxiter;
double dx, dy;

int main() {
    NX = 1000; NY = 400; Maxiter = 10; NUM_Points = 14000000;
    GRID_X = NX + 1; GRID_Y = NY + 1;
    dx = 1.0 / NX; dy = 1.0 / NY;

    double *mesh_value = NULL;
    Points *points = NULL;
    posix_memalign((void**)&mesh_value, 64, GRID_X * GRID_Y * sizeof(double));
    posix_memalign((void**)&points, 64, (size_t)NUM_Points * sizeof(Points));

    memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));
    memset(points, 0, (size_t)NUM_Points * sizeof(Points));
    initializepoints(points);

    printf("Iter, Interp_Time, Mover_Serial_Time, Total_Time\n");

    for (int iter = 0; iter < Maxiter; iter++) {
        clock_t start_i = clock();
        interpolation(mesh_value, points);
        clock_t end_i = clock();

        clock_t start_m = clock();
        mover_serial(points, dx, dy);
        clock_t end_m = clock();

        double t_i = (double)(end_i - start_i) / CLOCKS_PER_SEC;
        double t_m = (double)(end_m - start_m) / CLOCKS_PER_SEC;
        printf("%d, %lf, %lf, %lf\n", iter + 1, t_i, t_m, t_i + t_m);
    }
    free(mesh_value); free(points);
    return 0;
}