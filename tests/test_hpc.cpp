#include "particles.hpp"
#include "morton.hpp"
#include "backend.hpp"
#include "snapshot.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
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

bool test_morton_encoding_decoding() {
    BoundingBox bbox{-100.0f, -100.0f, -100.0f, 100.0f, 100.0f, 100.0f};

    // Test origin
    uint64_t m_orig = morton_encode_3d(0.0f, 0.0f, 0.0f, bbox);
    uint32_t qx, qy, qz;
    morton_decode_3d(m_orig, qx, qy, qz);
    ASSERT_TRUE(qx > 0 && qy > 0 && qz > 0, "Quantized coordinates should be centered around mid-grid");

    // Test invertibility of bit expansion/compaction
    for (uint32_t val : {0u, 1u, 42u, 1024u, 2097151u}) {
        uint64_t expanded = expand_bits_21(val);
        uint32_t compacted = compact_bits_21(expanded);
        ASSERT_TRUE(val == compacted, "expand_bits_21 and compact_bits_21 must be exact inverses");
    }

    // Test spatial ordering: particles along diagonal should have monotonically increasing Morton keys
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

bool test_bh_vs_direct_accuracy() {
    ParticleSystem ps;
    ps.init_disk(256, 50.0f, 500.0f, 100.0f);

    const float G = 1.0f;
    const float eps_sq = 0.04f;
    const float theta = 0.6f;

    auto backend = create_compute_backend();

    // Direct O(N^2)
    backend->direct_compute_forces(ps, G, eps_sq);
    std::vector<float> direct_ax = ps.ax;
    std::vector<float> direct_ay = ps.ay;
    std::vector<float> direct_az = ps.az;

    // Barnes-Hut
    backend->compute_forces(ps, theta, G, eps_sq);

    double sum_rel_err = 0.0;
    for (size_t i = 0; i < ps.count; i++) {
        double d_mag = std::sqrt(direct_ax[i]*direct_ax[i] + direct_ay[i]*direct_ay[i] + direct_az[i]*direct_az[i]);
        double diff_x = ps.ax[i] - direct_ax[i];
        double diff_y = ps.ay[i] - direct_ay[i];
        double diff_z = ps.az[i] - direct_az[i];
        double diff_mag = std::sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);

        double rel_err = (d_mag > 1e-8) ? (diff_mag / d_mag) : diff_mag;
        sum_rel_err += rel_err;
    }

    double mean_rel_err = (sum_rel_err / ps.count);
    ASSERT_TRUE(mean_rel_err < 0.02, "Barnes-Hut mean relative force error must be < 2%");
    return true;
}

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

int main() {
    std::cout << "=================================================\n";
    std::cout << "         AstroHPC C++ Engine & Morton Tests      \n";
    std::cout << "=================================================\n";

    RUN_TEST(test_morton_encoding_decoding);
    RUN_TEST(test_morton_sorting_locality);
    RUN_TEST(test_two_body_kepler_orbit);
    RUN_TEST(test_bh_vs_direct_accuracy);
    RUN_TEST(test_symplectic_reversibility);
    RUN_TEST(test_snapshot_roundtrip);

    std::cout << "=================================================\n";
    std::cout << "Result: " << tests_passed << " / " << tests_run
              << " tests passed (" << (tests_passed * 100 / tests_run) << "%)\n";
    std::cout << "=================================================\n";

    return (tests_passed == tests_run) ? 0 : 1;
}
