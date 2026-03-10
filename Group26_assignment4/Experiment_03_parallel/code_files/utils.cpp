#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

// Cloud-In-Cell (Bilinear) Interpolation (Serial Code)
void interpolation(double *mesh_value, Points *points) {
    
    // Reset mesh array for the current iteration
    memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

    for (int p = 0; p < NUM_Points; p++) {
        double px = points[p].x;
        double py = points[p].y;

        // Determine the cell index for the bottom-left node
        int i = (int)(px / dx);
        int j = (int)(py / dy);

        // Coordinates of the bottom-left node
        double x_i = i * dx;
        double y_j = j * dy;

        // Calculate fractional distances (weights)
        double hx = (px - x_i) / dx;
        double hy = (py - y_j) / dy;

        // Distribute charge/mass to the 4 surrounding nodes
        mesh_value[j * GRID_X + i] += (1.0 - hx) * (1.0 - hy);
        mesh_value[j * GRID_X + (i + 1)] += hx * (1.0 - hy);
        mesh_value[(j + 1) * GRID_X + i] += (1.0 - hx) * hy;
        mesh_value[(j + 1) * GRID_X + (i + 1)] += hx * hy;
    }
}

// Stochastic Mover (Serial Code) 
void mover_serial(Points *points, double deltaX, double deltaY) {
    for (int p = 0; p < NUM_Points; p++) {
        double new_x, new_y;
        
        do {
            double r_x = ((double)rand() / RAND_MAX) * 2.0 * deltaX - deltaX;
            double r_y = ((double)rand() / RAND_MAX) * 2.0 * deltaY - deltaY;
            
            new_x = points[p].x + r_x;
            new_y = points[p].y + r_y;
        } while (new_x < 0.0 || new_x >= 1.0 || new_y < 0.0 || new_y >= 1.0);
        
        points[p].x = new_x;
        points[p].y = new_y;
    }
}

// Stochastic Mover (Parallel Code) 
void mover_parallel(Points *points, double deltaX, double deltaY) {
    
    #pragma omp parallel
    {
        // Thread-safe random seed to prevent global locking overhead
        unsigned int seed = 12345 + omp_get_thread_num();
        
        // Static schedule chunks the array, preventing false sharing on 64-byte cache lines
        #pragma omp for schedule(static)
        for (int p = 0; p < NUM_Points; p++) {
            double new_x, new_y;
            
            do {
                double r_x = ((double)rand_r(&seed) / RAND_MAX) * 2.0 * deltaX - deltaX;
                double r_y = ((double)rand_r(&seed) / RAND_MAX) * 2.0 * deltaY - deltaY;
                
                new_x = points[p].x + r_x;
                new_y = points[p].y + r_y;
            } while (new_x < 0.0 || new_x >= 1.0 || new_y < 0.0 || new_y >= 1.0);
            
            points[p].x = new_x;
            points[p].y = new_y;
        }
    }
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