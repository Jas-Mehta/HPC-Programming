#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY, NUM_Points, Maxiter;
double dx, dy;

int main() {
    int grids_nx[] = {250, 500, 1000};
    int grids_ny[] = {100, 200, 400};
    Maxiter = 10;
    NUM_Points = 100000000;

    printf("Problem_Index, Grid_Size, Total_Interp_Time\n");

    for (int g = 0; g < 3; g++) {
        NX = grids_nx[g]; NY = grids_ny[g];
        GRID_X = NX + 1; GRID_Y = NY + 1;
        dx = 1.0 / NX; dy = 1.0 / NY;

        double *mesh_value = NULL;
        Points *points = NULL;
        posix_memalign((void**)&mesh_value, 64, GRID_X * GRID_Y * sizeof(double));
        if (posix_memalign((void**)&points, 64, (size_t)NUM_Points * sizeof(Points)) != 0 || !points) continue;

        memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));
        memset(points, 0, (size_t)NUM_Points * sizeof(Points));
        initializepoints(points);

        double total_interp_time = 0.0;
        for (int iter = 0; iter < Maxiter; iter++) {
            clock_t start = clock();
            interpolation(mesh_value, points);
            clock_t end = clock();
            total_interp_time += (double)(end - start) / CLOCKS_PER_SEC;
        }
        printf("%d, %dx%d, %lf\n", g + 1, NX, NY, total_interp_time);
        free(points);
        free(mesh_value);
    }
    return 0;
}