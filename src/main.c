#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "config.h"
#include "particles.h"
#include "octree.h"

// Statically allocated system and octree arena (Zero per-frame heap allocation)
static Particles sys;
static OctreePool pool;

static void print_banner(void) {
    printf("\033[38;5;99m               __             __               \033[0m\n");
    printf("\033[38;5;75m  ____ _____  / /__________  / /_  ____  _____ \033[0m\n");
    printf("\033[38;5;69m / __ `/ ___// __/ ___/ __ \\/ __ \\/ __ \\/ ___/ \033[0m\n");
    printf("\033[38;5;39m/ /_/ (__  )/ /_/ /  / /_/ / / / / /_/ / /__   \033[0m\n");
    printf("\033[38;5;38m\\__,_/____/ \\__/_/   \\____/_/ /_/ .___/\\___/   \033[0m\n");
    printf("\033[38;5;37m                               /_/             \033[0m\n\n");
}

static void print_usage(const char *prog) {
    print_banner();
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -n <int>        Number of bodies [1..%d] (default: 8192)\n", MAX_BODIES);
    printf("  -s <int>        Number of steps (default: 100)\n");
    printf("  -t <float>      MAC theta parameter (default: %.2f)\n", DEFAULT_THETA);
    printf("  -e <float>      Softening parameter epsilon^2 (default: %.2f)\n", DEFAULT_EPSILON_SQ);
    printf("  -d <float>      Time step dt (default: %.3f)\n", DEFAULT_DT);
    printf("  --preset <type> Preset: 'disk' or 'three_body' (default: disk)\n");
    printf("  --direct        Use direct O(N^2) computation\n");
    printf("  --compare       Compare Barnes-Hut vs Direct force on step 0\n");
    printf("  -b, --bench     Print per-phase benchmark timings\n");
    printf("  -h, --help      Display this help\n\n");
}

int main(int argc, char **argv) {
    int n = 8192;
    int steps = 100;
    float theta = DEFAULT_THETA;
    float eps_sq = DEFAULT_EPSILON_SQ;
    float dt = DEFAULT_DT;
    int use_direct = 0;
    int compare_mode = 0;
    int bench_mode = 0;
    const char *preset = "disk";

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            theta = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) {
            eps_sq = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            dt = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--preset") == 0 && i + 1 < argc) {
            preset = argv[++i];
        } else if (strcmp(argv[i], "--direct") == 0) {
            use_direct = 1;
        } else if (strcmp(argv[i], "--compare") == 0) {
            compare_mode = 1;
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--bench") == 0) {
            bench_mode = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (strcmp(preset, "three_body") == 0) {
        n = 3;
        eps_sq = 1e-2f;
    }

    if (n > MAX_BODIES) {
        fprintf(stderr, "Error: n=%d exceeds MAX_BODIES (%d)\n", n, MAX_BODIES);
        return 1;
    }

    print_banner();
    printf("astrohpc: bodies=%d, steps=%d, algo=%s, dt=%.4f, threads=%d\n\n",
           n, steps, use_direct ? "direct" : "barnes-hut", dt, omp_get_max_threads());

    // Initialize particles
    if (strcmp(preset, "three_body") == 0) {
        particles_init_three_body(&sys);
    } else {
        particles_init_disk(&sys, n, 100.0f, 1000.0f, 200.0f);
    }

    // Comparison mode: evaluate Barnes-Hut accuracy vs Direct summation
    if (compare_mode) {
        printf("--- Running Baseline Comparison (Step 0) ---\n");

        double t0 = omp_get_wtime();
        direct_compute_forces(&sys, n, DEFAULT_G, eps_sq);
        double t_direct = omp_get_wtime() - t0;

        float *direct_ax = malloc(n * sizeof(float));
        float *direct_ay = malloc(n * sizeof(float));
        float *direct_az = malloc(n * sizeof(float));
        memcpy(direct_ax, sys.ax, n * sizeof(float));
        memcpy(direct_ay, sys.ay, n * sizeof(float));
        memcpy(direct_az, sys.az, n * sizeof(float));

        double t1 = omp_get_wtime();
        octree_build(&pool, &sys, n);
        double t_tree_build = omp_get_wtime() - t1;

        double t2 = omp_get_wtime();
        octree_compute_forces(&pool, &sys, n, theta, DEFAULT_G, eps_sq);
        double t_bh_force = omp_get_wtime() - t2;

        double max_rel_err = 0.0;
        double sum_rel_err = 0.0;
        for (int i = 0; i < n; i++) {
            double d_mag = sqrt((double)direct_ax[i]*direct_ax[i] + (double)direct_ay[i]*direct_ay[i] + (double)direct_az[i]*direct_az[i]);
            double diff_x = sys.ax[i] - direct_ax[i];
            double diff_y = sys.ay[i] - direct_ay[i];
            double diff_z = sys.az[i] - direct_az[i];
            double diff_mag = sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);

            double rel_err = (d_mag > 1e-8) ? (diff_mag / d_mag) : diff_mag;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
            sum_rel_err += rel_err;
        }

        printf("  Direct O(N^2) Time:      %8.3f ms\n", t_direct * 1000.0);
        printf("  Barnes-Hut Total Time:   %8.3f ms (Build: %.2f ms, Force: %.2f ms)\n",
               (t_tree_build + t_bh_force) * 1000.0, t_tree_build * 1000.0, t_bh_force * 1000.0);
        printf("  Speedup:                 %8.2fx\n", t_direct / (t_tree_build + t_bh_force));
        printf("  Mean Relative Force Err: %8.4f%%\n", (sum_rel_err / n) * 100.0);
        printf("  Max  Relative Force Err: %8.4f%%\n", max_rel_err * 100.0);
        printf("  Octree Nodes Allocated:  %d / %d\n", pool.node_count, MAX_OCTREE_NODES);
        printf("--------------------------------------------\n\n");

        free(direct_ax);
        free(direct_ay);
        free(direct_az);
    }

    // Initial Energy Diagnostics
    if (n <= 4096) {
        double ke0, pe0;
        particles_compute_energy(&sys, n, DEFAULT_G, eps_sq, &ke0, &pe0);
        printf("Initial Energy -> KE: %11.4e | PE: %11.4e | Total: %11.4e\n\n", ke0, pe0, ke0 + pe0);
    }

    double total_sim_time = 0.0;
    double total_tree_time = 0.0;
    double total_force_time = 0.0;
    double total_integ_time = 0.0;

    printf("Simulating %d steps...\n", steps);

    for (int step = 0; step < steps; step++) {
        double step_start = omp_get_wtime();

        if (use_direct) {
            double t0 = omp_get_wtime();
            direct_compute_forces(&sys, n, DEFAULT_G, eps_sq);
            total_force_time += (omp_get_wtime() - t0);
        } else {
            // Pipeline Stage 1: Arena Reset & Stage 2: Octree Build
            double t0 = omp_get_wtime();
            octree_build(&pool, &sys, n);
            total_tree_time += (omp_get_wtime() - t0);

            // Pipeline Stage 3: Force Compute (Multipole Traversal)
            double t1 = omp_get_wtime();
            octree_compute_forces(&pool, &sys, n, theta, DEFAULT_G, eps_sq);
            total_force_time += (omp_get_wtime() - t1);
        }

        // Pipeline Stage 4: Symplectic Integration
        double t2 = omp_get_wtime();
        particles_integrate_symplectic(&sys, n, dt);
        total_integ_time += (omp_get_wtime() - t2);

        double step_dt = omp_get_wtime() - step_start;
        total_sim_time += step_dt;

        // Progress logging
        if (steps <= 20 || step % (steps / 10 == 0 ? 1 : steps / 10) == 0 || step == steps - 1) {
            if (strcmp(preset, "three_body") == 0) {
                printf("Step %3d | Body 1 Pos: (%6.2f, %6.2f) | Body 2 Pos: (%6.2f, %6.2f)\n",
                       step, sys.x[1], sys.y[1], sys.x[2], sys.y[2]);
            } else {
                printf("Step %4d / %d | Step Time: %6.2f ms | Instant FPS: %6.1f | Active Nodes: %5d\n",
                       step + 1, steps, step_dt * 1000.0, 1.0 / step_dt, pool.node_count);
            }
        }
    }

    printf("\nSummary:\n");
    printf("  Total time: %.3f s\n", total_sim_time);
    printf("  Time/step:  %.3f ms (%.1f FPS)\n",
           (total_sim_time / steps) * 1000.0, (double)steps / total_sim_time);

    if (bench_mode) {
        printf("\nBreakdown:\n");
        if (!use_direct) {
            printf("  Octree build:   %8.3f ms/step (%5.1f%%)\n",
                   (total_tree_time / steps) * 1000.0, (total_tree_time / total_sim_time) * 100.0);
        }
        printf("  Force compute:  %8.3f ms/step (%5.1f%%)\n",
               (total_force_time / steps) * 1000.0, (total_force_time / total_sim_time) * 100.0);
        printf("  Integration:    %8.3f ms/step (%5.1f%%)\n",
               (total_integ_time / steps) * 1000.0, (total_integ_time / total_sim_time) * 100.0);

        double pairwise_equivalent = (double)n * (double)n * (double)steps;
        printf("  Interactions:   ~%.2e pairs/sec equivalent\n", pairwise_equivalent / total_sim_time);
    }

    if (n <= 4096) {
        double ke1, pe1;
        particles_compute_energy(&sys, n, DEFAULT_G, eps_sq, &ke1, &pe1);
        printf("\nEnergy: KE=%.4e | PE=%.4e | Total=%.4e\n", ke1, pe1, ke1 + pe1);
    }

    return 0;
}
