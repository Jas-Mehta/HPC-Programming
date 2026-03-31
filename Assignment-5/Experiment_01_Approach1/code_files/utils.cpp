#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

void interpolation(double * __restrict__ mesh_value, Points * __restrict__ points) {
    memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

    for (int p = 0; p < NUM_Points; p++) {
        double px = points[p].x;
        double py = points[p].y;

        int i = (int)(px / dx);
        int j = (int)(py / dy);

        double x_i = i * dx;
        double y_j = j * dy;

        double hx = (px - x_i) / dx;
        double hy = (py - y_j) / dy;

        mesh_value[j * GRID_X + i] += (1.0 - hx) * (1.0 - hy);
        mesh_value[j * GRID_X + (i + 1)] += hx * (1.0 - hy);
        mesh_value[(j + 1) * GRID_X + i] += (1.0 - hx) * hy;
        mesh_value[(j + 1) * GRID_X + (i + 1)] += hx * hy;
    }
}

void mover_serial_deferred(Points * __restrict__ points, double deltaX, double deltaY) {
    for (int p = 0; p < NUM_Points; p++) {
        double r_x = ((double)rand() / RAND_MAX) * 2.0 * deltaX - deltaX;
        double r_y = ((double)rand() / RAND_MAX) * 2.0 * deltaY - deltaY;
        points[p].x += r_x;
        points[p].y += r_y;
    }

    int left = 0;
    int right = NUM_Points - 1;

    while (left <= right) {
        bool left_valid = (points[left].x >= 0.0 && points[left].x < 1.0 && points[left].y >= 0.0 && points[left].y < 1.0);
        if (left_valid) {
            left++;
        } else {
            bool right_valid = (points[right].x >= 0.0 && points[right].x < 1.0 && points[right].y >= 0.0 && points[right].y < 1.0);
            if (!right_valid) {
                right--;
            } else {
                Points temp = points[left];
                points[left] = points[right];
                points[right] = temp;
                left++;
                right--;
            }
        }
    }

    for (int p = left; p < NUM_Points; p++) {
        points[p].x = (double)rand() / RAND_MAX;
        points[p].y = (double)rand() / RAND_MAX;
    }
}

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
