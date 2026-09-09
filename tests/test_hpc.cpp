#include "particles.hpp"
#include "morton.hpp"
#include "backend.hpp"
#include "snapshot.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <cassert>

using namespace astro;

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "\n  [FAIL] " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        return false; \
    } \
} while (0)

#define RUN_TEST(test_func) do { \
    tests_run++; \
    std::cout << "  [RUN] " << #test_func << "..." << std::flush; \
    if (test_func()) { \
        tests_passed++; \
        std::cout << "  [PASS] " << #test_func << "\n"; \
    } else { \
        std::cout << "  [FAIL] " << #test_func << "\n"; \
    } \
} while (0)

// 1. 64-bit Morton encoding and decoding invertibility
bool test_morton_encoding_decoding() {
    BoundingBox bbox{-100.0f, -100.0f, -100.0f, 100.0f, 100.0f, 100.0f};

    uint64_t m_orig = morton_encode_3d(0.0f, 0.0f, 0.0f, bbox);
    uint32_t qx, qy, qz;
    morton_decode_3d(m_orig, qx, qy, qz);
    ASSERT_TRUE(qx > 0 && qy > 0 && qz > 0, "Quantized coordinates should be centered around mid-grid");

    for (uint32_t val : {0u, 1u, 42u, 1024u, 2097151u}) {
        uint64_t expanded = expand_bits_21(val);
        uint32_t compacted = compact_bits_21(expanded);
        ASSERT_TRUE(val == compacted, "expand_bits_21 and compact_bits_21 must be exact inverses");
    }

    ParticleSystem ps;
    ps.resize(5);
    for (size_t i = 0; i < 5; i++) {
        float coord = -50.0f + (float)i * 25.0f;
        ps.x[i] = coord;
        ps.y[i] = coord;
        ps.z[i] = coord;
    }
    ps.compute_morton_keys();
    for (size_t i = 1; i < 5; i++) {
        ASSERT_TRUE(ps.morton[i] > ps.morton[i - 1], "Morton keys along space diagonal must be monotonically increasing");
    }

    return true;
}

// 2. Morton spatial sorting
bool test_morton_sorting_locality() {
    ParticleSystem ps;
    ps.resize(100);
    for (size_t i = 0; i < 100; i++) {
        ps.x[i] = (float)(rand() % 200 - 100);
        ps.y[i] = (float)(rand() % 200 - 100);
        ps.z[i] = (float)(rand() % 200 - 100);
        ps.m[i] = 1.0f + (float)i;
    }

    ps.sort_by_morton();

    for (size_t i = 1; i < ps.count; i++) {
        ASSERT_TRUE(ps.morton[i] >= ps.morton[i - 1], "Particles must be sorted in non-decreasing Morton key order");
    }
    return true;
}

// 3. Kepler two-body orbit baseline
bool test_two_body_kepler_orbit() {
    ParticleSystem ps;
    ps.resize(2);
    ps.reset_accelerations();

    const float m1 = 1000.0f;
    const float m2 = 1.0f;
    const float r = 10.0f;
    const float G = 1.0f;
    const float v_circ = std::sqrt(G * m1 / r);

    ps.x[0] = 0.0f; ps.y[0] = 0.0f; ps.z[0] = 0.0f;
    ps.vx[0] = 0.0f; ps.vy[0] = 0.0f; ps.vz[0] = 0.0f;
    ps.m[0] = m1;

    ps.x[1] = r; ps.y[1] = 0.0f; ps.z[1] = 0.0f;
    ps.vx[1] = 0.0f; ps.vy[1] = v_circ; ps.vz[1] = 0.0f;
    ps.m[1] = m2;

    const float eps_sq = 1e-4f;
    const float dt = 0.005f;
    const int steps = 500;

    auto backend = create_compute_backend();
    ASSERT_TRUE(backend != nullptr, "Backend factory must produce valid engine");

    double ke0, pe0;
    ps.compute_energy(G, eps_sq, ke0, pe0);
    const double e0 = ke0 + pe0;

    for (int s = 0; s < steps; s++) {
        backend->compute_forces(ps, 0.5f, G, eps_sq);
        ps.integrate_symplectic(dt);
    }

    double ke_final, pe_final;
    ps.compute_energy(G, eps_sq, ke_final, pe_final);
    const double e_final = ke_final + pe_final;

    double rel_error = std::abs((e_final - e0) / e0);
    ASSERT_TRUE(rel_error < 1e-3, "Kepler 2-body energy relative error must be < 0.1%");
    return true;
}

// 4. Long-duration energy stability test (2000 steps)
bool test_long_duration_energy_drift() {
    ParticleSystem ps;
    ps.resize(2);
    ps.reset_accelerations();

    const float m1 = 500.0f;
    const float m2 = 2.0f;
    const float r = 12.0f;
    const float G = 1.0f;
    const float v_circ = std::sqrt(G * (m1 + m2) / r);

    ps.x[0] = 0.0f; ps.y[0] = 0.0f; ps.z[0] = 0.0f;
    ps.vx[0] = 0.0f; ps.vy[0] = -v_circ * (m2 / (m1 + m2)); ps.vz[0] = 0.0f;
    ps.m[0] = m1;

    ps.x[1] = r; ps.y[1] = 0.0f; ps.z[1] = 0.0f;
    ps.vx[1] = 0.0f; ps.vy[1] = v_circ * (m1 / (m1 + m2)); ps.vz[1] = 0.0f;
    ps.m[1] = m2;

    const float eps_sq = 1e-4f;
    const float dt = 0.005f;
    const int steps = 2000;

    auto backend = create_compute_backend();

    double ke0, pe0;
    ps.compute_energy(G, eps_sq, ke0, pe0);
    const double e0 = ke0 + pe0;

    for (int s = 0; s < steps; s++) {
        backend->compute_forces(ps, 0.5f, G, eps_sq);
        ps.integrate_symplectic(dt);
    }

    double ke_f, pe_f;
    ps.compute_energy(G, eps_sq, ke_f, pe_f);
    double rel_error = std::abs((ke_f + pe_f - e0) / e0);

    ASSERT_TRUE(rel_error < 0.005, "Long-term (2000 steps) energy drift must remain bounded (< 0.5%)");
    return true;
}

// 5. Numerical convergence test across timestep sizes
bool test_timestep_convergence() {
    auto run_sim = [](float dt) -> double {
        ParticleSystem ps;
        ps.resize(2);
        ps.reset_accelerations();
        const float m1 = 1000.0f, m2 = 1.0f, r = 10.0f, G = 1.0f;
        ps.x[0] = 0; ps.y[0] = 0; ps.z[0] = 0;
        ps.vx[0] = 0; ps.vy[0] = 0; ps.vz[0] = 0;
        ps.m[0] = m1;
        ps.x[1] = r; ps.y[1] = 0; ps.z[1] = 0;
        ps.vx[1] = 0; ps.vy[1] = std::sqrt(G * m1 / r); ps.vz[1] = 0;
        ps.m[1] = m2;

        const float eps_sq = 1e-4f;
        const int steps = static_cast<int>(2.0f / dt);

        auto backend = create_compute_backend();
        double ke0, pe0;
        ps.compute_energy(G, eps_sq, ke0, pe0);
        double e0 = ke0 + pe0;

        for (int s = 0; s < steps; s++) {
            backend->compute_forces(ps, 0.4f, G, eps_sq);
            ps.integrate_symplectic(dt);
        }

        double ke_f, pe_f;
        ps.compute_energy(G, eps_sq, ke_f, pe_f);
        return std::abs((ke_f + pe_f - e0) / e0);
    };

    double err_large = run_sim(0.02f);
    double err_medium = run_sim(0.01f);
    double err_small = run_sim(0.005f);

    ASSERT_TRUE(err_small < err_medium, "Error at dt=0.005 must be less than dt=0.01");
    ASSERT_TRUE(err_medium < err_large, "Error at dt=0.01 must be less than dt=0.02");
    return true;
}

// 6. Linear and angular momentum conservation test
bool test_linear_and_angular_momentum_conservation() {
    ParticleSystem ps;
    ps.init_three_body();

    double px0, py0, pz0;
    ps.compute_momentum(px0, py0, pz0);

    double lx0, ly0, lz0;
    ps.compute_angular_momentum(lx0, ly0, lz0);

    auto backend = create_compute_backend();
    const float G = 1.0f;
    const float eps_sq = 0.01f;
    const float dt = 0.001f;

    for (int s = 0; s < 100; s++) {
        backend->direct_compute_forces(ps, G, eps_sq);
        ps.integrate_symplectic(dt);
    }

    double px_f, py_f, pz_f;
    ps.compute_momentum(px_f, py_f, pz_f);

    double lx_f, ly_f, lz_f;
    ps.compute_angular_momentum(lx_f, ly_f, lz_f);

    double delta_p = std::sqrt((px_f - px0)*(px_f - px0) + (py_f - py0)*(py_f - py0) + (pz_f - pz0)*(pz_f - pz0));
    double delta_l = std::sqrt((lx_f - lx0)*(lx_f - lx0) + (ly_f - ly0)*(ly_f - ly0) + (lz_f - lz0)*(lz_f - lz0));

    ASSERT_TRUE(delta_p < 1e-4, "Linear momentum must be conserved in isolated system");
    ASSERT_TRUE(delta_l < 1e-4, "Angular momentum must be conserved in isolated system");
    return true;
}

// 7. Barnes-Hut accuracy sweep across multiple theta values
bool test_bh_accuracy_and_theta_sweep() {
    ParticleSystem ps;
    ps.init_disk(256, 50.0f, 500.0f, 100.0f);

    const float G = 1.0f;
    const float eps_sq = 0.04f;

    auto backend = create_compute_backend();

    // Direct O(N^2) baseline
    backend->direct_compute_forces(ps, G, eps_sq);
    std::vector<float> d_ax = ps.ax;
    std::vector<float> d_ay = ps.ay;
    std::vector<float> d_az = ps.az;

    auto evaluate_error = [&](float theta) -> double {
        backend->compute_forces(ps, theta, G, eps_sq);
        double sum_err = 0.0;
        for (size_t i = 0; i < ps.count; i++) {
            double d_mag = std::sqrt(d_ax[i]*d_ax[i] + d_ay[i]*d_ay[i] + d_az[i]*d_az[i]);
            double dx = ps.ax[i] - d_ax[i];
            double dy = ps.ay[i] - d_ay[i];
            double dz = ps.az[i] - d_az[i];
            double diff = std::sqrt(dx*dx + dy*dy + dz*dz);
            sum_err += (d_mag > 1e-6) ? (diff / d_mag) : diff;
        }
        return sum_err / ps.count;
    };

    double err_fine = evaluate_error(0.3f);
    double err_mid  = evaluate_error(0.6f);
    double err_coarse = evaluate_error(0.9f);

    ASSERT_TRUE(err_fine < 0.01, "Theta=0.3 mean relative force error must be < 1%");
    ASSERT_TRUE(err_mid < 0.03, "Theta=0.6 mean relative force error must be < 3%");
    ASSERT_TRUE(err_fine <= err_mid, "Smaller theta must yield strictly lower force error");
    ASSERT_TRUE(err_mid <= err_coarse, "Mid theta must yield lower or equal error than coarse theta");
    return true;
}

// 8. Symplectic reversibility test
bool test_symplectic_reversibility() {
    ParticleSystem ps;
    ps.init_disk(64, 40.0f, 300.0f, 50.0f);

    std::vector<float> orig_x = ps.x;
    std::vector<float> orig_y = ps.y;
    std::vector<float> orig_z = ps.z;

    const float G = 1.0f;
    const float eps_sq = 0.04f;
    const float dt = 0.01f;
    const int steps = 50;

    auto backend = create_compute_backend();

    // Forward
    for (int s = 0; s < steps; s++) {
        backend->compute_forces(ps, 0.65f, G, eps_sq);
        ps.integrate_symplectic(dt);
    }

    // Backward using adjoint symplectic steps
    for (int s = 0; s < steps; s++) {
        ps.integrate_symplectic_reverse_pos(dt);
        backend->compute_forces(ps, 0.65f, G, eps_sq);
        ps.integrate_symplectic_reverse_vel(dt);
    }

    float max_drift = 0.0f;
    for (size_t i = 0; i < ps.count; i++) {
        float dx = ps.x[i] - orig_x[i];
        float dy = ps.y[i] - orig_y[i];
        float dz = ps.z[i] - orig_z[i];
        float drift = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (drift > max_drift) max_drift = drift;
    }

    ASSERT_TRUE(max_drift < 1e-5f, "Symplectic integration must be exactly reversible within floating point precision");
    return true;
}

// 9. Snapshot roundtrip test
bool test_snapshot_roundtrip() {
    ParticleSystem ps1;
    ps1.init_disk(128, 60.0f, 400.0f, 80.0f);

    const std::string tmp_file = "/tmp/test_snapshot_hpc.bin";
    int ret = write_snapshot(tmp_file, ps1, 42, 1.25f, 0.01f);
    ASSERT_TRUE(ret == 0, "Snapshot write must succeed");

    ParticleSystem ps2;
    SnapshotHeader hdr;
    bool ok = read_snapshot(tmp_file, ps2, &hdr);
    ASSERT_TRUE(ok, "Snapshot read must succeed");
    ASSERT_TRUE(hdr.n == 128, "Snapshot body count must match");
    ASSERT_TRUE(hdr.step == 42, "Snapshot step must match");

    for (size_t i = 0; i < ps1.count; i++) {
        ASSERT_TRUE(ps1.x[i] == ps2.x[i], "Position X must match bit-for-bit");
        ASSERT_TRUE(ps1.y[i] == ps2.y[i], "Position Y must match bit-for-bit");
        ASSERT_TRUE(ps1.z[i] == ps2.z[i], "Position Z must match bit-for-bit");
        ASSERT_TRUE(ps1.m[i] == ps2.m[i], "Mass must match bit-for-bit");
    }

    return true;
}

// 10. Pathological Octree: Coincident / Near-identical particles (Mass Conservation & No Dropping)
bool test_pathological_coincident_particles() {
    ParticleSystem ps;
    const int n_coincident = 16;
    ps.resize(n_coincident + 1);
    ps.reset_accelerations();

    // 16 particles at the exact same point (5.0, 5.0, 5.0)
    for (int i = 0; i < n_coincident; i++) {
        ps.x[i] = 5.0f;
        ps.y[i] = 5.0f;
        ps.z[i] = 5.0f;
        ps.m[i] = 10.0f;
    }

    // 1 test body at (15.0, 5.0, 5.0)
    ps.x[n_coincident] = 15.0f;
    ps.y[n_coincident] = 5.0f;
    ps.z[n_coincident] = 5.0f;
    ps.m[n_coincident] = 1.0f;

    auto backend = create_compute_backend();
    const float G = 1.0f;
    const float eps_sq = 1e-4f;

    // Barnes-Hut force evaluation must not hang, must not drop particles, and must handle coincident cluster
    backend->compute_forces(ps, 0.5f, G, eps_sq);

    // Theoretical acceleration on the test body at distance dx = 10:
    // a = -G * (16 * 10.0) / (dx^2 + eps_sq) = -160.0 / 100.0001 ~ -1.6
    float ax_test = ps.ax[n_coincident];
    ASSERT_TRUE(std::abs(ax_test - (-1.6f)) < 0.05f, "Coincident particles must combine mass exactly without drop");
    return true;
}

// 11. Pathological Octree: Extreme scales and zero-mass test particles
bool test_pathological_extreme_scales() {
    auto backend = create_compute_backend();

    // Test Micro Scale (10^-6)
    {
        ParticleSystem ps;
        ps.resize(4);
        ps.reset_accelerations();
        for (size_t i = 0; i < 4; i++) {
            ps.x[i] = 1e-6f * (float)i;
            ps.y[i] = 1e-6f * (float)i;
            ps.z[i] = 1e-6f * (float)i;
            ps.m[i] = 1e-12f;
        }
        backend->compute_forces(ps, 0.5f, 1.0f, 1e-14f);
        for (size_t i = 0; i < 4; i++) {
            ASSERT_TRUE(!std::isnan(ps.ax[i]) && !std::isinf(ps.ax[i]), "Micro-scale coordinates must not produce NaN or Inf");
        }
    }

    // Test Macro Scale (10^7)
    {
        ParticleSystem ps;
        ps.resize(4);
        ps.reset_accelerations();
        for (size_t i = 0; i < 4; i++) {
            ps.x[i] = 1e7f * (float)i;
            ps.y[i] = 1e7f * (float)i;
            ps.z[i] = 1e7f * (float)i;
            ps.m[i] = 1e6f;
        }
        backend->compute_forces(ps, 0.5f, 1.0f, 1.0f);
        for (size_t i = 0; i < 4; i++) {
            ASSERT_TRUE(!std::isnan(ps.ax[i]) && !std::isinf(ps.ax[i]), "Macro-scale coordinates must not produce NaN or Inf");
        }
    }

    // Test Zero-mass particles (test bodies)
    {
        ParticleSystem ps;
        ps.resize(3);
        ps.reset_accelerations();
        ps.x[0] = 0.0f; ps.y[0] = 0.0f; ps.z[0] = 0.0f; ps.m[0] = 100.0f;
        ps.x[1] = 5.0f; ps.y[1] = 0.0f; ps.z[1] = 0.0f; ps.m[1] = 0.0f; // Zero mass
        ps.x[2] = -5.0f; ps.y[2] = 0.0f; ps.z[2] = 0.0f; ps.m[2] = 0.0f; // Zero mass

        backend->compute_forces(ps, 0.5f, 1.0f, 0.01f);
        // Central body should experience ~0 force (mass 0 bodies don't pull)
        ASSERT_TRUE(std::abs(ps.ax[0]) < 1e-4f, "Zero-mass bodies must exert zero gravitational force");
        // Test bodies must feel pull from central body
        ASSERT_TRUE(ps.ax[1] < -1.0f, "Zero-mass test body must feel gravitational pull from massive body");
    }

    return true;
}

// 12. Performance regression threshold
bool test_performance_regression() {
    ParticleSystem ps;
    ps.init_disk(1024, 60.0f, 600.0f, 100.0f);

    auto backend = create_compute_backend();
    const float G = 1.0f;
    const float eps_sq = 0.04f;
    const float theta = 0.65f;

    auto t0 = std::chrono::high_resolution_clock::now();
    const int bench_steps = 5;
    for (int s = 0; s < bench_steps; s++) {
        backend->compute_forces(ps, theta, G, eps_sq);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_steps;

    std::cout << " (avg " << std::fixed << std::setprecision(2) << avg_ms << " ms/step for N=1024) ";
    ASSERT_TRUE(avg_ms < 40.0, "Performance regression: 1024 bodies must evaluate in < 40 ms/step");
    return true;
}

// 13. Coarse subtree extraction & conservation (Distributed LET building block)
bool test_coarse_tree_extraction_and_conservation() {
    ParticleSystem ps;
    ps.init_disk(512, 50.0f, 500.0f, 150.0f);

    float total_mass = 0.0f;
    double true_com_x = 0.0, true_com_y = 0.0, true_com_z = 0.0;
    for (size_t i = 0; i < ps.count; i++) {
        total_mass += ps.m[i];
        true_com_x += ps.m[i] * ps.x[i];
        true_com_y += ps.m[i] * ps.y[i];
        true_com_z += ps.m[i] * ps.z[i];
    }
    true_com_x /= total_mass;
    true_com_y /= total_mass;
    true_com_z /= total_mass;

    auto backend = create_compute_backend();
    backend->compute_forces(ps, 0.65f, 1.0f, 0.04f);

    std::vector<RemoteMultipole> coarse_nodes;
    backend->extract_coarse_nodes(2, 0, coarse_nodes);

    ASSERT_TRUE(!coarse_nodes.empty(), "Extracted coarse nodes list should not be empty");
    ASSERT_TRUE(coarse_nodes.size() <= 64, "At depth <= 2, coarse cut should have <= 64 cells");

    float coarse_total_mass = 0.0f;
    double coarse_com_x = 0.0, coarse_com_y = 0.0, coarse_com_z = 0.0;
    for (const auto& node : coarse_nodes) {
        ASSERT_TRUE(node.mass > 0.0f, "Coarse node mass must be positive");
        ASSERT_TRUE(node.half_size > 0.0f, "Coarse node half_size must be positive");
        coarse_total_mass += node.mass;
        coarse_com_x += node.mass * node.com_x;
        coarse_com_y += node.mass * node.com_y;
        coarse_com_z += node.mass * node.com_z;
    }
    coarse_com_x /= coarse_total_mass;
    coarse_com_y /= coarse_total_mass;
    coarse_com_z /= coarse_total_mass;

    float mass_err = std::abs(coarse_total_mass - total_mass) / total_mass;
    ASSERT_TRUE(mass_err < 1e-4f, "Coarse nodes total mass must match true particle mass");

    double com_dist = std::sqrt((coarse_com_x - true_com_x) * (coarse_com_x - true_com_x) +
                                (coarse_com_y - true_com_y) * (coarse_com_y - true_com_y) +
                                (coarse_com_z - true_com_z) * (coarse_com_z - true_com_z));
    ASSERT_TRUE(com_dist < 1e-2, "Coarse nodes aggregate COM must match system COM");

    return true;
}

int main() {
    std::cout << "=================================================\n";
    std::cout << "         AstroHPC Comprehensive Test Suite       \n";
    std::cout << "=================================================\n";

    RUN_TEST(test_morton_encoding_decoding);
    RUN_TEST(test_morton_sorting_locality);
    RUN_TEST(test_two_body_kepler_orbit);
    RUN_TEST(test_long_duration_energy_drift);
    RUN_TEST(test_timestep_convergence);
    RUN_TEST(test_linear_and_angular_momentum_conservation);
    RUN_TEST(test_bh_accuracy_and_theta_sweep);
    RUN_TEST(test_symplectic_reversibility);
    RUN_TEST(test_snapshot_roundtrip);
    RUN_TEST(test_pathological_coincident_particles);
    RUN_TEST(test_pathological_extreme_scales);
    RUN_TEST(test_performance_regression);
    RUN_TEST(test_coarse_tree_extraction_and_conservation);

    std::cout << "=================================================\n";
    std::cout << "Result: " << tests_passed << " / " << tests_run
              << " tests passed (" << (tests_passed * 100 / tests_run) << "%)\n";
    std::cout << "=================================================\n";

    return (tests_passed == tests_run) ? 0 : 1;
}
