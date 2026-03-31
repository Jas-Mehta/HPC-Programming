#ifndef UTILS_H
#define UTILS_H
#include <time.h>
#include "init.h"

void interpolation(double * __restrict__ mesh_value, Points * __restrict__ points);
void mover_serial_immediate(Points * __restrict__ points, double deltaX, double deltaY);
void save_mesh(double *mesh_value);

#endif
