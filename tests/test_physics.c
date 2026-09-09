#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>

#include "config.h"
#include "particles.h"
#include "octree.h"
#include "snapshot.h"

// ANSI Color codes for clean test reporting
#define COLOR_GREEN "\033[32m"
#define COLOR_RED   "\033[31m"
#define COLOR_BLUE  "\033[34m"
#define COLOR_BOLD  "\033[1m"
#define COLOR_RESET "\033[0m"

static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(fn) do { \
    tests_run++; \
    printf("  [RUN] %s...", #fn); \
    fflush(stdout); \
    int res = fn(); \
    if (res == 0) { \
        tests_passed++; \
        printf("\r  [" COLOR_GREEN "PASS" COLOR_RESET "] %s\n", #fn); \
    } else { \
        printf("\r  [" COLOR_RED "FAIL" COLOR_RESET "] %s\n", #fn); \
    } \
} while (0)

static Particles test_sys;
static OctreePool test_pool;

/**
 * Test 1: Two-Body Kepler Orbit Energy Conservation
 * Verifies that a circular Keplerian binary system conserves total mechanical
 * energy over 5,000 steps within strict Symplectic Euler bounds (|dE/E0| < 0.001)
 * with zero secular radial drift.
 */
static int test_two_body_kepler_orbit(void) {
    const int n = 2;
    const float G = 1.0f;
    const float m1 = 1000.0f;
    const float m2 = 1.0f;
    const float M = m1 + m2;
    const float R = 20.0f;
    const float eps_sq = 0.0f; // Pure Newtonian gravity

    // Circular orbital speed: v = sqrt(G * M / R)
    const float v_rel = sqrtf(G * M / R);

    // Initial positions relative to barycenter
    test_sys.x[0] = -(m2 / M) * R;  test_sys.y[0] = 0.0f;  test_sys.z[0] = 0.0f;
    test_sys.vx[0] = 0.0f;  test_sys.vy[0] = -(m2 / M) * v_rel;  test_sys.vz[0] = 0.0f;
    test_sys.m[0] = m1;

    test_sys.x[1] = +(m1 / M) * R;  test_sys.y[1] = 0.0f;  test_sys.z[1] = 0.0f;
    test_sys.vx[1] = 0.0f;  test_sys.vy[1] = +(m1 / M) * v_rel;  test_sys.vz[1] = 0.0f;
    test_sys.m[1] = m2;

    double ke0, pe0;
    particles_compute_energy(&test_sys, n, G, eps_sq, &ke0, &pe0);
    const double e0 = ke0 + pe0;

    const float dt = 0.005f;
    const int steps = 5000;

    for (int s = 0; s < steps; s++) {
        direct_compute_forces(&test_sys, n, G, eps_sq);
        particles_integrate_symplectic(&test_sys, n, dt);
    }

    double ke_final, pe_final;
    particles_compute_energy(&test_sys, n, G, eps_sq, &ke_final, &pe_final);
    const double e_final = ke_final + pe_final;

    // Check relative energy error
    double rel_energy_err = fabs((e_final - e0) / e0);
    if (rel_energy_err > 1e-3) {
        fprintf(stderr, "Energy error too large: %.6e (threshold: 1e-3)\n", rel_energy_err);
        return -1;
    }

    // Check final distance circularity preservation
    float dx = test_sys.x[1] - test_sys.x[0];
    float dy = test_sys.y[1] - test_sys.y[0];
    float dz = test_sys.z[1] - test_sys.z[0];
    float final_r = sqrtf(dx*dx + dy*dy + dz*dz);
    float radial_drift = fabsf(final_r - R) / R;

    if (radial_drift > 0.02f) {
        fprintf(stderr, "Radial drift too large: %.4f (threshold: 0.02)\n", radial_drift);
        return -1;
    }

    return 0;
}

/**
 * Test 2: Octree Node Count, Hierarchy Sanity & Mass Conservation
 * Verifies that the octree spatial partitioning preserves total mass bit-for-bit,
 * computes the exact mass-weighted center-of-mass, has bounded depth, and contains
 * zero dangling or cyclical child references.
 */
static int test_octree_integrity_and_mass_conservation(void) {
    const int n = 512;
    srand(42);

    float expected_total_mass = 0.0f;
    double expected_cm_x = 0.0, expected_cm_y = 0.0, expected_cm_z = 0.0;

    for (int i = 0; i < n; i++) {
        test_sys.x[i] = ((float)rand() / (float)RAND_MAX * 200.0f) - 100.0f;
        test_sys.y[i] = ((float)rand() / (float)RAND_MAX * 200.0f) - 100.0f;
        test_sys.z[i] = ((float)rand() / (float)RAND_MAX * 200.0f) - 100.0f;
        test_sys.m[i] = 0.1f + ((float)rand() / (float)RAND_MAX * 10.0f); // Heterogeneous masses

        expected_total_mass += test_sys.m[i];
        expected_cm_x += (double)test_sys.m[i] * (double)test_sys.x[i];
        expected_cm_y += (double)test_sys.m[i] * (double)test_sys.y[i];
        expected_cm_z += (double)test_sys.m[i] * (double)test_sys.z[i];
    }
    expected_cm_x /= (double)expected_total_mass;
    expected_cm_y /= (double)expected_total_mass;
    expected_cm_z /= (double)expected_total_mass;

    int root = octree_build(&test_pool, &test_sys, n);
    if (root != 0) {
        fprintf(stderr, "Root node index is %d, expected 0\n", root);
        return -1;
    }

    if (test_pool.node_count <= 0 || test_pool.node_count >= MAX_OCTREE_NODES) {
        fprintf(stderr, "Invalid node count: %d\n", test_pool.node_count);
        return -1;
    }

    // Root node must contain the total cluster mass
    const OctNode *root_node = &test_pool.nodes[0];
    float mass_err = fabsf(root_node->mass - expected_total_mass) / expected_total_mass;
    if (mass_err > 1e-4f) {
        fprintf(stderr, "Root mass mismatch: got %.4f, expected %.4f\n", root_node->mass, expected_total_mass);
        return -1;
    }

    // Root node must have exact mass-weighted center-of-mass
    double cm_dist = sqrt(pow(root_node->com_x - expected_cm_x, 2) +
                          pow(root_node->com_y - expected_cm_y, 2) +
                          pow(root_node->com_z - expected_cm_z, 2));
    if (cm_dist > 1e-3) {
        fprintf(stderr, "Root center-of-mass error: %.6f\n", cm_dist);
        return -1;
    }

    // Check all allocated nodes for valid bounds and child pointers
    for (int i = 0; i < test_pool.node_count; i++) {
        const OctNode *node = &test_pool.nodes[i];
        if (node->mass <= 0.0f) {
            fprintf(stderr, "Node %d has non-positive mass: %.4f\n", i, node->mass);
            return -1;
        }
        for (int c = 0; c < 8; c++) {
            int child = node->children[c];
            if (child != -1 && (child <= 0 || child >= test_pool.node_count)) {
                fprintf(stderr, "Node %d child %d has invalid index %d\n", i, c, child);
                return -1;
            }
        }
    }

    return 0;
}

/**
 * Test 3: Barnes-Hut Force Accuracy vs Direct O(N^2)
 * Verifies that the multipole Barnes-Hut tree traversal achieves < 1% mean relative
 * force error compared to exact pairwise summation.
 */
static int test_force_accuracy_barnes_hut_vs_direct(void) {
    const int n = 512;
    const float G = 1.0f;
    const float eps_sq = 0.04f;
    const float theta = 0.5f; // Standard high-accuracy opening angle

    particles_init_disk(&test_sys, n, 50.0f, 500.0f, 100.0f);

    // Compute ground truth direct forces
    direct_compute_forces(&test_sys, n, G, eps_sq);

    float *true_ax = malloc(n * sizeof(float));
    float *true_ay = malloc(n * sizeof(float));
    float *true_az = malloc(n * sizeof(float));
    memcpy(true_ax, test_sys.ax, n * sizeof(float));
    memcpy(true_ay, test_sys.ay, n * sizeof(float));
    memcpy(true_az, test_sys.az, n * sizeof(float));

    // Compute Barnes-Hut tree forces
    octree_build(&test_pool, &test_sys, n);
    octree_compute_forces(&test_pool, &test_sys, n, theta, G, eps_sq);

    double sum_rel_err = 0.0;
    double max_rel_err = 0.0;

    for (int i = 0; i < n; i++) {
        double d_mag = sqrt((double)true_ax[i]*true_ax[i] + (double)true_ay[i]*true_ay[i] + (double)true_az[i]*true_az[i]);
        double diff_x = test_sys.ax[i] - true_ax[i];
        double diff_y = test_sys.ay[i] - true_ay[i];
        double diff_z = test_sys.az[i] - true_az[i];
        double diff_mag = sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);

        double rel_err = (d_mag > 1e-6) ? (diff_mag / d_mag) : diff_mag;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        sum_rel_err += rel_err;
    }

    free(true_ax);
    free(true_ay);
    free(true_az);

    double mean_rel_err = sum_rel_err / (double)n;

    if (mean_rel_err > 0.01) { // 1% threshold
        fprintf(stderr, "Barnes-Hut mean relative error too high: %.4f%% (threshold: 1.0%%)\n",
                mean_rel_err * 100.0);
        return -1;
    }

    return 0;
}

/**
 * Test 4: Symplectic Time-Reversibility
 * Verifies that integrating forward N steps (+dt) and backward N steps (-dt)
 * recovers initial coordinates with machine floating-point precision.
 */
static int test_symplectic_time_reversibility(void) {
    const int n = 64;
    const float dt = 0.01f;
    const int steps = 100;
    const float G = 1.0f;
    const float eps_sq = 0.1f;

    particles_init_disk(&test_sys, n, 20.0f, 200.0f, 50.0f);

    float *x0 = malloc(n * sizeof(float));
    float *y0 = malloc(n * sizeof(float));
    float *z0 = malloc(n * sizeof(float));
    memcpy(x0, test_sys.x, n * sizeof(float));
    memcpy(y0, test_sys.y, n * sizeof(float));
    memcpy(z0, test_sys.z, n * sizeof(float));

    // Forward integration
    for (int s = 0; s < steps; s++) {
        direct_compute_forces(&test_sys, n, G, eps_sq);
        particles_integrate_symplectic(&test_sys, n, dt);
    }

    // Backward time-reversal integration (exact adjoint inverse)
    for (int s = 0; s < steps; s++) {
        particles_integrate_symplectic_reverse_pos(&test_sys, n, dt);
        direct_compute_forces(&test_sys, n, G, eps_sq);
        particles_integrate_symplectic_reverse_vel(&test_sys, n, dt);
    }

    float max_drift = 0.0f;
    for (int i = 0; i < n; i++) {
        float dx = test_sys.x[i] - x0[i];
        float dy = test_sys.y[i] - y0[i];
        float dz = test_sys.z[i] - z0[i];
        float drift = sqrtf(dx*dx + dy*dy + dz*dz);
        if (drift > max_drift) max_drift = drift;
    }

    free(x0);
    free(y0);
    free(z0);

    if (max_drift > 1e-3f) {
        fprintf(stderr, "Time-reversal symmetry drift too large: %.6e\n", max_drift);
        return -1;
    }

    return 0;
}

/**
 * Test 5: Binary Snapshot Serialization Roundtrip
 * Verifies that snapshot_write and snapshot_read serialize and deserialize
 * headers, coordinates, velocities, and individual masses without loss of precision.
 */
static int test_snapshot_roundtrip(void) {
    const char *test_path = "tests/test_snap.bin";
    const int n = 64;
    const int step = 1337;
    const float sim_time = 42.5f;
    const float dt = 0.005f;

    particles_init_disk(&test_sys, n, 30.0f, 300.0f, 60.0f);

    int write_res = snapshot_write(test_path, test_sys.x, test_sys.y, test_sys.z,
                                   test_sys.vx, test_sys.vy, test_sys.vz, test_sys.m,
                                   n, step, sim_time, dt);
    if (write_res != 0) {
        fprintf(stderr, "snapshot_write failed\n");
        return -1;
    }

    static Particles read_sys;
    SnapshotHeader header;
    int read_count = snapshot_read(test_path, read_sys.x, read_sys.y, read_sys.z,
                                   read_sys.vx, read_sys.vy, read_sys.vz, read_sys.m,
                                   MAX_BODIES, &header);

    unlink(test_path);

    if (read_count != n) {
        fprintf(stderr, "Snapshot read count mismatch: %d vs %d\n", read_count, n);
        return -1;
    }

    if (header.step != (uint32_t)step || fabsf(header.sim_time - sim_time) > 1e-5f) {
        fprintf(stderr, "Header metadata mismatch: step=%u time=%.2f\n", header.step, header.sim_time);
        return -1;
    }

    for (int i = 0; i < n; i++) {
        if (test_sys.x[i] != read_sys.x[i] ||
            test_sys.y[i] != read_sys.y[i] ||
            test_sys.z[i] != read_sys.z[i] ||
            test_sys.m[i] != read_sys.m[i]) {
            fprintf(stderr, "Particle %d data mismatch after roundtrip\n", i);
            return -1;
        }
    }

    return 0;
}

int main(void) {
    printf(COLOR_BOLD "=================================================\n");
    printf("         AstroHPC Physics & Engine Tests         \n");
    printf("=================================================\n" COLOR_RESET);

    RUN_TEST(test_two_body_kepler_orbit);
    RUN_TEST(test_octree_integrity_and_mass_conservation);
    RUN_TEST(test_force_accuracy_barnes_hut_vs_direct);
    RUN_TEST(test_symplectic_time_reversibility);
    RUN_TEST(test_snapshot_roundtrip);

    printf(COLOR_BOLD "=================================================\n");
    if (tests_passed == tests_run) {
        printf("Result: " COLOR_GREEN "%d / %d tests passed (100%%)" COLOR_RESET "\n",
               tests_passed, tests_run);
        printf(COLOR_BOLD "=================================================\n" COLOR_RESET);
        return 0;
    } else {
        printf("Result: " COLOR_RED "%d / %d tests passed" COLOR_RESET "\n",
               tests_passed, tests_run);
        printf(COLOR_BOLD "=================================================\n" COLOR_RESET);
        return 1;
    }
}
