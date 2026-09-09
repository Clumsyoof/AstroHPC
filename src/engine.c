#include "engine.h"
#include "config.h"
#include "particles.h"
#include "octree.h"

#include <string.h>
#include <omp.h>

static Particles g_sys;
static OctreePool g_pool;

static int g_n = 0;
static int g_step = 0;
static double g_last_step_time = 0.0;
static float g_last_dt = DEFAULT_DT;
static float g_last_theta = DEFAULT_THETA;
static float g_last_eps_sq = DEFAULT_EPSILON_SQ;

void sim_init_disk(int n, float radius, float central_mass, float disk_mass) {
    if (n > MAX_BODIES) n = MAX_BODIES;
    g_n = n;
    g_step = 0;
    g_last_step_time = 0.0;
    particles_init_disk(&g_sys, n, radius, central_mass, disk_mass);
    octree_build(&g_pool, &g_sys, g_n);
}

void sim_init_three_body(void) {
    g_n = 3;
    g_step = 0;
    g_last_step_time = 0.0;
    particles_init_three_body(&g_sys);
    octree_build(&g_pool, &g_sys, g_n);
}

void sim_step(float dt, float theta, float eps_sq) {
    if (g_n <= 0) return;

    g_last_dt = dt;
    g_last_theta = theta;
    g_last_eps_sq = eps_sq;

    double t0 = omp_get_wtime();

    // 1. Octree Build
    octree_build(&g_pool, &g_sys, g_n);

    // 2. Barnes-Hut multipole force traversal
    octree_compute_forces(&g_pool, &g_sys, g_n, theta, DEFAULT_G, eps_sq);

    // 3. Symplectic Integration
    particles_integrate_symplectic(&g_sys, g_n, dt);

    g_last_step_time = omp_get_wtime() - t0;
    g_step++;
}

SimStats sim_get_stats(void) {
    SimStats stats;
    memset(&stats, 0, sizeof(stats));

    stats.n = g_n;
    stats.step = g_step;
    stats.active_nodes = g_pool.node_count;
    stats.dt = g_last_dt;
    stats.theta = g_last_theta;
    stats.eps_sq = g_last_eps_sq;
    stats.step_time_ms = g_last_step_time * 1000.0;
    stats.fps = (g_last_step_time > 1e-6) ? (1.0 / g_last_step_time) : 0.0;

    if (g_n > 0 && g_n <= 4096) {
        particles_compute_energy(&g_sys, g_n, DEFAULT_G, g_last_eps_sq,
                                 &stats.kinetic_energy, &stats.potential_energy);
        stats.total_energy = stats.kinetic_energy + stats.potential_energy;
    }

    return stats;
}

int sim_get_positions_2d(float *out_x, float *out_y, int max_count) {
    if (!out_x || !out_y || max_count <= 0) return 0;
    int count = (g_n < max_count) ? g_n : max_count;
    for (int i = 0; i < count; i++) {
        out_x[i] = g_sys.x[i];
        out_y[i] = g_sys.y[i];
    }
    return count;
}
