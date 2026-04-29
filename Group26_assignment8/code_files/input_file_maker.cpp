#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * Generates a binary input file for the MPI+OpenMP PIC code.
 *
 * File format:
 *   int  NX, NY          – grid dimensions (cells, not points)
 *   int  NUM_Points       – number of particles
 *   int  Maxiter          – number of iterations
 *   NUM_Points * (double x, double y)  – initial particle positions in [0,1]
 *
 * Particles are written ONCE.  The mover evolves their positions across
 * iterations, so there is no need to write Maxiter copies.
 *
 * Usage:
 *   Interactive:  ./input_maker.out
 *   Non-interactive (for scripts):
 *     ./input_maker.out <NX> <NY> <NUM_Points> <Maxiter> [output_file]
 */

static void generate(int NX, int NY, int NUM_Points, int Maxiter,
                     const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: cannot create %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fwrite(&NX,         sizeof(int), 1, fp);
    fwrite(&NY,         sizeof(int), 1, fp);
    fwrite(&NUM_Points, sizeof(int), 1, fp);
    fwrite(&Maxiter,    sizeof(int), 1, fp);

    srand((unsigned int)time(NULL));

    for (int i = 0; i < NUM_Points; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        fwrite(&x, sizeof(double), 1, fp);
        fwrite(&y, sizeof(double), 1, fp);
    }

    fclose(fp);
    printf("Generated '%s'  (NX=%d NY=%d points=%d iters=%d)\n",
           filename, NX, NY, NUM_Points, Maxiter);
}

int main(int argc, char **argv) {
    int  NX, NY, NUM_Points, Maxiter;
    char filename[256] = "input.bin";

    if (argc >= 5) {
        NX         = atoi(argv[1]);
        NY         = atoi(argv[2]);
        NUM_Points = atoi(argv[3]);
        Maxiter    = atoi(argv[4]);
        if (argc >= 6) snprintf(filename, sizeof(filename), "%s", argv[5]);
    } else {
        printf("Enter grid dimensions (NX NY): ");
        scanf("%d %d", &NX, &NY);
        printf("Enter number of particles: ");
        scanf("%d", &NUM_Points);
        printf("Enter number of iterations: ");
        scanf("%d", &Maxiter);
        printf("Enter output filename [input.bin]: ");
        char tmp[256];
        if (scanf("%255s", tmp) == 1 && tmp[0] != '\0')
            snprintf(filename, sizeof(filename), "%s", tmp);
    }

    generate(NX, NY, NUM_Points, Maxiter, filename);
    return 0;
}
