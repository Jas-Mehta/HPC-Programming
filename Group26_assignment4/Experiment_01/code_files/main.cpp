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
    int points_range[] = {100, 10000, 1000000, 100000000, 1000000000};
    Maxiter = 10;

    for (int g = 0; g < 3; g++) {
        NX = grids_nx[g]; NY = grids_ny[g];
        GRID_X = NX + 1; GRID_Y = NY + 1;
        dx = 1.0 / NX; dy = 1.0 / NY;

        double *mesh_value = NULL;
        posix_memalign((void**)&mesh_value, 64, GRID_X * GRID_Y * sizeof(double));

        printf("Grid Configuration: %d x %d\n", NX, NY);
        printf("Particles, Total_Interp_Time\n");

        for (int p = 0; p < 5; p++) {
            NUM_Points = points_range[p];
            Points *points = NULL;
            if (posix_memalign((void**)&points, 64, (size_t)NUM_Points * sizeof(Points)) != 0 || !points) continue;
            memset(points, 0, (size_t)NUM_Points * sizeof(Points));

            double total_interp_time = 0.0;
            for (int iter = 0; iter < Maxiter; iter++) {
                initializepoints(points);
                clock_t start = clock();
                interpolation(mesh_value, points);
                clock_t end = clock();
                total_interp_time += (double)(end - start) / CLOCKS_PER_SEC;
            }
            printf("%d, %lf\n", NUM_Points, total_interp_time);
            free(points);
        }
        free(mesh_value);
    }
    return 0;
}