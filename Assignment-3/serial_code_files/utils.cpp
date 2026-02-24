#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

// Naive serial interpolation (baseline)
void interpolation(double *mesh_value, Points *points) {
    double inv_dx = 1.0 / dx;
    double inv_dy = 1.0 / dy;
    double cell_area = dx * dy;

    for (int k = 0; k < NUM_Points; k++) {
        double sx = points[k].x * inv_dx;
        double sy = points[k].y * inv_dy;
        int col = (int)sx;
        int row = (int)sy;
        if (col >= NX) col = NX - 1;
        if (row >= NY) row = NY - 1;
        double fx = sx - col;
        double fy = sy - row;
        double ifx = 1.0 - fx;
        double ify = 1.0 - fy;
        int base = row * GRID_X + col;
        mesh_value[base]              += ifx * ify * cell_area;
        mesh_value[base + 1]          += fx  * ify * cell_area;
        mesh_value[base + GRID_X]     += ifx * fy  * cell_area;
        mesh_value[base + GRID_X + 1] += fx  * fy  * cell_area;
    }
}

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
