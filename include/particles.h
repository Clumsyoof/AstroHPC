#ifndef PARTICLES_H
#define PARTICLES_H

#include "config.h"

// Structure of Arrays (SoA) layout aligned to 64-byte boundaries for AVX/SIMD
typedef struct {
    ALIGN64 float x[MAX_BODIES];
    ALIGN64 float y[MAX_BODIES];
    ALIGN64 float z[MAX_BODIES];
    ALIGN64 float vx[MAX_BODIES];
    ALIGN64 float vy[MAX_BODIES];
    ALIGN64 float vz[MAX_BODIES];
    ALIGN64 float ax[MAX_BODIES];
    ALIGN64 float ay[MAX_BODIES];
    ALIGN64 float az[MAX_BODIES];
    ALIGN64 float m[MAX_BODIES];
} Particles;

// Presets and Loaders
void particles_init_three_body(Particles *sys);
void particles_init_disk(Particles *sys, int n, float radius, float central_mass, float total_disk_mass);
int particles_load_csv(Particles *sys, const char *filepath, int max_bodies);

// Clear accelerations before force computation
void particles_reset_accelerations(Particles *sys, int n);

// Symplectic (Semi-implicit) Euler integration step
void particles_integrate_symplectic(Particles *sys, int n, float dt);

// Diagnostic energy computation (Kinetic + Potential)
void particles_compute_energy(const Particles *sys, int n, float G, float eps_sq, double *kinetic, double *potential);

#endif // PARTICLES_H
