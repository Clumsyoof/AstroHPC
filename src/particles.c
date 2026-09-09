#include "particles.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void particles_reset_accelerations(Particles *sys, int n) {
    memset(sys->ax, 0, (size_t)n * sizeof(float));
    memset(sys->ay, 0, (size_t)n * sizeof(float));
    memset(sys->az, 0, (size_t)n * sizeof(float));
}

void particles_init_three_body(Particles *sys) {
    particles_reset_accelerations(sys, 3);

    // Central mass
    sys->x[0] = 0.0f;  sys->y[0] = 0.0f;  sys->z[0] = 0.0f;
    sys->vx[0] = 0.0f; sys->vy[0] = 0.0f; sys->vz[0] = 0.0f;
    sys->m[0] = 1000.0f;

    // Body 1
    sys->x[1] = 5.0f;  sys->y[1] = 0.0f;  sys->z[1] = 0.0f;
    sys->vx[1] = 0.0f; sys->vy[1] = 14.0f; sys->vz[1] = 0.0f;
    sys->m[1] = 1.0f;

    // Body 2
    sys->x[2] = 10.0f; sys->y[2] = 0.0f;  sys->z[2] = 0.0f;
    sys->vx[2] = 0.0f; sys->vy[2] = 10.0f; sys->vz[2] = 0.0f;
    sys->m[2] = 1.0f;
}

void particles_init_disk(Particles *sys, int n, float radius, float central_mass, float total_disk_mass) {
    particles_reset_accelerations(sys, n);

    if (n <= 0) return;

    // Deterministic seed for reproducible testing
    srand(42);

    // Central supermassive body
    sys->x[0] = 0.0f;
    sys->y[0] = 0.0f;
    sys->z[0] = 0.0f;
    sys->vx[0] = 0.0f;
    sys->vy[0] = 0.0f;
    sys->vz[0] = 0.0f;
    sys->m[0] = central_mass;

    if (n == 1) return;

    const float disk_particle_mass = total_disk_mass / (float)(n - 1);
    const float min_r = radius * 0.05f;

    for (int i = 1; i < n; i++) {
        // Power-law radial distribution for disk density
        const float u = (float)rand() / (float)RAND_MAX;
        const float r = min_r + (radius - min_r) * sqrtf(u);

        const float theta = 2.0f * (float)M_PI * ((float)rand() / (float)RAND_MAX);
        const float cos_t = cosf(theta);
        const float sin_t = sinf(theta);

        // Thin disk vertical scale height
        const float z_scale = radius * 0.02f;
        const float z_offset = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * z_scale;

        sys->x[i] = r * cos_t;
        sys->y[i] = r * sin_t;
        sys->z[i] = z_offset;
        sys->m[i] = disk_particle_mass;

        // Circular orbital velocity v = sqrt(G * M_enclosed / r)
        // Approximate enclosed mass = central_mass + fraction of disk mass
        const float m_enclosed = central_mass + total_disk_mass * (r / radius);
        const float v_circ = sqrtf((DEFAULT_G * m_enclosed) / r);

        // Tangential velocity vector (-sin, cos) + small random thermal dispersion
        const float disp = v_circ * 0.03f;
        const float vx_disp = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * disp;
        const float vy_disp = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * disp;
        const float vz_disp = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * disp;

        sys->vx[i] = -v_circ * sin_t + vx_disp;
        sys->vy[i] =  v_circ * cos_t + vy_disp;
        sys->vz[i] = vz_disp;
    }
}

void particles_integrate_symplectic(Particles *sys, int n, float dt) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        // Step 1: Update velocity using current acceleration
        sys->vx[i] += sys->ax[i] * dt;
        sys->vy[i] += sys->ay[i] * dt;
        sys->vz[i] += sys->az[i] * dt;

        // Step 2: Update position using new velocity (Symplectic Euler)
        sys->x[i] += sys->vx[i] * dt;
        sys->y[i] += sys->vy[i] * dt;
        sys->z[i] += sys->vz[i] * dt;
    }
}

void particles_compute_energy(const Particles *sys, int n, float G, float eps_sq, double *kinetic, double *potential) {
    double total_ke = 0.0;
    double total_pe = 0.0;

    #pragma omp parallel for reduction(+:total_ke) schedule(static)
    for (int i = 0; i < n; i++) {
        const double v_sq = (double)sys->vx[i] * sys->vx[i] +
                            (double)sys->vy[i] * sys->vy[i] +
                            (double)sys->vz[i] * sys->vz[i];
        total_ke += 0.5 * (double)sys->m[i] * v_sq;
    }

    #pragma omp parallel for reduction(+:total_pe) schedule(guided)
    for (int i = 0; i < n; i++) {
        const float xi = sys->x[i];
        const float yi = sys->y[i];
        const float zi = sys->z[i];
        const float mi = sys->m[i];

        for (int j = i + 1; j < n; j++) {
            const float dx = sys->x[j] - xi;
            const float dy = sys->y[j] - yi;
            const float dz = sys->z[j] - zi;
            const float dist = sqrtf(dx*dx + dy*dy + dz*dz + eps_sq);
            total_pe -= (double)(G * mi * sys->m[j]) / (double)dist;
        }
    }

    *kinetic = total_ke;
    *potential = total_pe;
}
