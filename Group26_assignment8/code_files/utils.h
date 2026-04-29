#ifndef UTILS_H
#define UTILS_H

#include "init.h"

// Forward interpolation: n_pts local particles → partial_mesh (pre-zeroed).
// Uses OpenMP privatisation internally; no race conditions.
void interpolation_local(double *partial_mesh, Points *points, int n_pts);

// Normalise mesh values to [-1, 1]; saves original fmin/fmax for denormalise.
void normalize_mesh(double *mesh, int gsz, double *out_fmin, double *out_fmax);

// Restore mesh from normalised [-1, 1] back to original scale.
void denormalize_mesh(double *mesh, int gsz, double fmin, double fmax);

// Reverse interpolation (mover): update local particle positions from mesh.
// Particles leaving [0,1]^2 are marked inactive.
void mover_local(double *mesh, Points *points, int n_pts);

// Write global mesh to Mesh.out.
void save_mesh(double *mesh);

#endif
