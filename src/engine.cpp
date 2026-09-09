#include "engine.h"
#include "config.h"
#include "particles.hpp"
#include "backend.hpp"

#include <chrono>
#include <cstring>
#include <memory>

static astro::ParticleSystem g_sys;
static std::unique_ptr<astro::IComputeBackend> g_backend;

static int g_n = 0;
static int g_step = 0;
static double g_last_step_time = 0.0;
static float g_last_dt = DEFAULT_DT;
static float g_last_theta = DEFAULT_THETA;
static float g_last_eps_sq = DEFAULT_EPSILON_SQ;

static void ensure_backend() {
    if (!g_backend) {
        g_backend = astro::create_compute_backend();
    }
}

extern "C" {

void sim_init_disk(int n, float radius, float central_mass, float disk_mass) {
    ensure_backend();
    g_n = n;
    g_step = 0;
    g_last_step_time = 0.0;
    g_sys.init_disk(static_cast<size_t>(n), radius, central_mass, disk_mass);
}

void sim_init_three_body(void) {
    ensure_backend();
    g_n = 3;
    g_step = 0;
    g_last_step_time = 0.0;
    g_sys.init_three_body();
}

void sim_step(float dt, float theta, float eps_sq) {
    if (g_n <= 0) return;
    ensure_backend();

    g_last_dt = dt;
    g_last_theta = theta;
    g_last_eps_sq = eps_sq;

    auto t0 = std::chrono::high_resolution_clock::now();

    g_backend->compute_forces(g_sys, theta, DEFAULT_G, eps_sq);
    g_sys.integrate_symplectic(dt);

    auto t1 = std::chrono::high_resolution_clock::now();
    g_last_step_time = std::chrono::duration<double>(t1 - t0).count();
    g_step++;
}

SimStats sim_get_stats(void) {
    SimStats stats;
    std::memset(&stats, 0, sizeof(stats));

    stats.n = g_n;
    stats.step = g_step;
    stats.active_nodes = static_cast<int>(g_sys.count);
    stats.dt = g_last_dt;
    stats.theta = g_last_theta;
    stats.eps_sq = g_last_eps_sq;
    stats.step_time_ms = g_last_step_time * 1000.0;
    stats.fps = (g_last_step_time > 1e-6) ? (1.0 / g_last_step_time) : 0.0;

    if (g_n > 0 && g_n <= 4096) {
        g_sys.compute_energy(DEFAULT_G, g_last_eps_sq, stats.kinetic_energy, stats.potential_energy);
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

} // extern "C"
