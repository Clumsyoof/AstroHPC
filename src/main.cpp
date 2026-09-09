#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

#if defined(ASTRO_ENABLE_OPENMP) || defined(_OPENMP)
#include <omp.h>
#endif

#include "config.h"
#include "particles.hpp"
#include "backend.hpp"
#include "snapshot.hpp"
#include "mpi_domain.hpp"

using namespace astro;

static void print_banner() {
    std::cout << "\033[38;5;99m               __             __               \033[0m\n"
              << "\033[38;5;75m  ____ _____  / /__________  / /_  ____  _____ \033[0m\n"
              << "\033[38;5;69m / __ `/ ___// __/ ___/ __ \\/ __ \\/ __ \\/ ___/ \033[0m\n"
              << "\033[38;5;39m/ /_/ (__  )/ /_/ /  / /_/ / / / / /_/ / /__   \033[0m\n"
              << "\033[38;5;38m\\__,_/____/ \\__/_/   \\____/_/ /_/ .___/\\___/   \033[0m\n"
              << "\033[38;5;37m                               /_/             \033[0m\n\n";
}

static void print_usage(const char* prog) {
    print_banner();
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -n <int>        Number of bodies (default: 8192)\n"
              << "  -s <int>        Number of steps (default: 100)\n"
              << "  -g, --grav <f>  Gravitational constant G (default: " << DEFAULT_G << ", IRL: " << G_IRL_ASTRO << ")\n"
              << "  --real, --irl   Use accurate real-world constants (G=" << G_IRL_ASTRO << ")\n"
              << "  -t <float>      MAC theta parameter (default: " << DEFAULT_THETA << ")\n"
              << "  -e <float>      Softening parameter epsilon^2 (default: " << DEFAULT_EPSILON_SQ << ")\n"
              << "  -d <float>      Time step dt (default: " << DEFAULT_DT << ")\n"
              << "  --preset <type> Preset: 'disk' or 'three_body' (default: disk)\n"
              << "  -f, --file <path> Load particles from CSV dataset (e.g. Gaia DR3)\n"
              << "  --direct        Use direct O(N^2) computation\n"
              << "  --compare       Compare Barnes-Hut vs Direct force on step 0\n"
              << "  --dump <dir>    Directory to dump binary snapshots\n"
              << "  --dump-interval <int> Snapshot step interval (default: 1)\n"
              << "  -b, --bench     Print per-phase benchmark timings\n"
              << "  -h, --help      Display this help\n\n";
}

int main(int argc, char** argv) {
    auto mpi = MpiContext::init(&argc, &argv);

    int n = 8192;
    int steps = 100;
    float g_val = DEFAULT_G;
    bool g_custom = false;
    float theta = DEFAULT_THETA;
    float eps_sq = DEFAULT_EPSILON_SQ;
    bool eps_custom = false;
    float dt = DEFAULT_DT;
    bool dt_custom = false;
    bool use_direct = false;
    bool compare_mode = false;
    bool bench_mode = false;
    std::string preset = "disk";
    std::string csv_file = "";
    std::string dump_dir = "";
    int dump_interval = 1;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            n = std::atoi(argv[++i]);
        } else if (arg == "-s" && i + 1 < argc) {
            steps = std::atoi(argv[++i]);
        } else if ((arg == "-g" || arg == "--g" || arg == "--grav") && i + 1 < argc) {
            g_val = std::stof(argv[++i]);
            g_custom = true;
        } else if (arg == "--real" || arg == "--irl") {
            g_val = G_IRL_ASTRO;
            g_custom = true;
        } else if (arg == "-t" && i + 1 < argc) {
            theta = std::stof(argv[++i]);
        } else if (arg == "-e" && i + 1 < argc) {
            eps_sq = std::stof(argv[++i]);
            eps_custom = true;
        } else if (arg == "-d" && i + 1 < argc) {
            dt = std::stof(argv[++i]);
            dt_custom = true;
        } else if (arg == "--preset" && i + 1 < argc) {
            preset = argv[++i];
        } else if ((arg == "-f" || arg == "--file") && i + 1 < argc) {
            csv_file = argv[++i];
        } else if (arg == "--direct") {
            use_direct = true;
        } else if (arg == "--compare") {
            compare_mode = true;
        } else if (arg == "--dump" && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (arg == "--dump-interval" && i + 1 < argc) {
            dump_interval = std::atoi(argv[++i]);
        } else if (arg == "-b" || arg == "--bench") {
            bench_mode = true;
        } else if (arg == "-h" || arg == "--help") {
            if (mpi.is_root()) print_usage(argv[0]);
            mpi.finalize();
            return 0;
        } else {
            if (mpi.is_root()) {
                std::cerr << "Unknown argument: " << arg << "\n";
                print_usage(argv[0]);
            }
            mpi.finalize();
            return 1;
        }
    }

    ParticleSystem ps;

    if (!csv_file.empty()) {
        if (!ps.load_csv(csv_file)) {
            if (mpi.is_root()) std::cerr << "Error: Failed to load CSV file: " << csv_file << "\n";
            mpi.finalize();
            return 1;
        }
        n = static_cast<int>(ps.count);
        if (!g_custom) {
            g_val = ps.default_g();
        }
        if (!eps_custom) {
            eps_sq = ps.default_eps_sq();
        }
        if (!dt_custom) {
            dt = ps.default_dt();
        }
    } else if (preset == "three_body") {
        ps.init_three_body();
        n = 3;
        eps_sq = 1e-2f;
    } else {
        ps.init_disk(n, 100.0f, 1000.0f, 200.0f);
    }

    // Partition initial particles across MPI ranks if distributed
    if (mpi.enabled) {
        mpi.partition_particles(ps);
    }

    auto backend = create_compute_backend();

    if (mpi.is_root()) {
        print_banner();
        std::cout << "astrohpc (C++ Modern HPC): backend=" << backend->name();
        if (mpi.enabled) {
            std::cout << " [MPI " << mpi.size << " ranks]";
        }
#if defined(ASTRO_ENABLE_OPENMP) || defined(_OPENMP)
        std::cout << " [OpenMP " << omp_get_max_threads() << " threads]";
#endif
        std::cout << ", bodies=" << n << ", steps=" << steps
                  << ", algo=" << (use_direct ? "direct" : "barnes-hut")
                  << ", G=" << g_val << ", dt=" << dt << "\n\n";
    }

    if (compare_mode) {
        if (mpi.enabled) {
            if (mpi.is_root()) {
                std::cout << "--- Running Distributed Baseline Comparison (Step 0) ---\n";
            }
            // 1. Evaluate Distributed LET (Local Tree + Remote Coarse Subtree Cut)
            auto t0_let = std::chrono::high_resolution_clock::now();

            BoundingBox g_box = mpi.global_bounding_box(ps);
            mpi.migrate_particles(ps, g_box);

            backend->compute_forces(ps, theta, g_val, eps_sq);

            std::vector<RemoteMultipole> local_coarse;
            backend->extract_coarse_nodes(2, mpi.rank, local_coarse);
            auto remote_multipoles = mpi.exchange_multipoles(local_coarse);

#if defined(ASTRO_ENABLE_OPENMP) || defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (size_t i = 0; i < ps.count; i++) {
                const float xi = ps.x[i];
                const float yi = ps.y[i];
                const float zi = ps.z[i];

                float d_ax = 0.0f;
                float d_ay = 0.0f;
                float d_az = 0.0f;

                for (const auto& rm : remote_multipoles) {
                    if (rm.rank == mpi.rank || rm.mass <= 0.0f) continue;
                    const float dx = rm.com_x - xi;
                    const float dy = rm.com_y - yi;
                    const float dz = rm.com_z - zi;
                    const float dist_sq = dx * dx + dy * dy + dz * dz + eps_sq;

                    const float inv_dist = 1.0f / std::sqrt(dist_sq);
                    const float inv_cube = inv_dist * inv_dist * inv_dist;
                    const float scale = g_val * rm.mass * inv_cube;

                    d_ax += scale * dx;
                    d_ay += scale * dy;
                    d_az += scale * dz;
                }

                ps.ax[i] += d_ax;
                ps.ay[i] += d_ay;
                ps.az[i] += d_az;
            }

            auto t1_let = std::chrono::high_resolution_clock::now();
            double t_let = std::chrono::duration<double, std::milli>(t1_let - t0_let).count();

            // Store distributed LET accelerations for error comparison
            std::vector<float> dist_ax = ps.ax;
            std::vector<float> dist_ay = ps.ay;
            std::vector<float> dist_az = ps.az;

            // 2. Compute exact ground-truth global all-pairs direct force across all MPI ranks
            auto t0_direct = std::chrono::high_resolution_clock::now();
            std::vector<int> displs;
            auto all_gp = mpi.gather_all_particles(ps, displs);
            size_t total_n = all_gp.size();

            double local_max_err = 0.0;
            double local_sum_err = 0.0;

#if defined(ASTRO_ENABLE_OPENMP) || defined(_OPENMP)
#pragma omp parallel
            {
                double thread_max_err = 0.0;
                double thread_sum_err = 0.0;

#pragma omp for schedule(static)
                for (size_t i = 0; i < ps.count; i++) {
                    const float xi = ps.x[i];
                    const float yi = ps.y[i];
                    const float zi = ps.z[i];
                    const size_t global_i = displs[mpi.rank] + i;

                    float true_ax = 0.0f;
                    float true_ay = 0.0f;
                    float true_az = 0.0f;

                    for (size_t j = 0; j < total_n; j++) {
                        if (j == global_i) continue;
                        const float dx = all_gp[j].x - xi;
                        const float dy = all_gp[j].y - yi;
                        const float dz = all_gp[j].z - zi;
                        const float dist_sq = dx * dx + dy * dy + dz * dz + eps_sq;

                        const float inv_dist = 1.0f / std::sqrt(dist_sq);
                        const float inv_cube = inv_dist * inv_dist * inv_dist;
                        const float scale = g_val * all_gp[j].m * inv_cube;

                        true_ax += scale * dx;
                        true_ay += scale * dy;
                        true_az += scale * dz;
                    }

                    double d_mag = std::sqrt(true_ax * true_ax + true_ay * true_ay + true_az * true_az);
                    double diff_x = dist_ax[i] - true_ax;
                    double diff_y = dist_ay[i] - true_ay;
                    double diff_z = dist_az[i] - true_az;
                    double diff_mag = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

                    double rel_err = (d_mag > 1e-8) ? (diff_mag / d_mag) : diff_mag;
                    if (rel_err > thread_max_err) thread_max_err = rel_err;
                    thread_sum_err += rel_err;
                }

#pragma omp critical
                {
                    if (thread_max_err > local_max_err) local_max_err = thread_max_err;
                    local_sum_err += thread_sum_err;
                }
            }
#else
            for (size_t i = 0; i < ps.count; i++) {
                const float xi = ps.x[i];
                const float yi = ps.y[i];
                const float zi = ps.z[i];
                const size_t global_i = displs[mpi.rank] + i;

                float true_ax = 0.0f;
                float true_ay = 0.0f;
                float true_az = 0.0f;

                for (size_t j = 0; j < total_n; j++) {
                    if (j == global_i) continue;
                    const float dx = all_gp[j].x - xi;
                    const float dy = all_gp[j].y - yi;
                    const float dz = all_gp[j].z - zi;
                    const float dist_sq = dx * dx + dy * dy + dz * dz + eps_sq;

                    const float inv_dist = 1.0f / std::sqrt(dist_sq);
                    const float inv_cube = inv_dist * inv_dist * inv_dist;
                    const float scale = g_val * all_gp[j].m * inv_cube;

                    true_ax += scale * dx;
                    true_ay += scale * dy;
                    true_az += scale * dz;
                }

                double d_mag = std::sqrt(true_ax * true_ax + true_ay * true_ay + true_az * true_az);
                double diff_x = dist_ax[i] - true_ax;
                double diff_y = dist_ay[i] - true_ay;
                double diff_z = dist_az[i] - true_az;
                double diff_mag = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

                double rel_err = (d_mag > 1e-8) ? (diff_mag / d_mag) : diff_mag;
                if (rel_err > local_max_err) local_max_err = rel_err;
                local_sum_err += rel_err;
            }
#endif

            auto t1_direct = std::chrono::high_resolution_clock::now();
            double t_direct = std::chrono::duration<double, std::milli>(t1_direct - t0_direct).count();

            double global_max_err = 0.0;
            double global_sum_err = 0.0;
            double global_direct_ms = 0.0;
            double global_let_ms = 0.0;

#ifdef ASTRO_ENABLE_MPI
            MPI_Reduce(&local_max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_sum_err, &global_sum_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&t_direct, &global_direct_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&t_let, &global_let_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#else
            global_max_err = local_max_err;
            global_sum_err = local_sum_err;
            global_direct_ms = t_direct;
            global_let_ms = t_let;
#endif

            if (mpi.is_root()) {
                std::cout << std::fixed << std::setprecision(3);
                std::cout << "  Global Particles:          " << total_n << " (across " << mpi.size << " ranks)\n";
                std::cout << "  Global Direct O(N^2) Time: " << std::setw(8) << global_direct_ms << " ms\n";
                std::cout << "  Distributed LET Time:      " << std::setw(8) << global_let_ms << " ms\n";
                std::cout << "  Speedup:                   " << std::setw(8) << (global_direct_ms / std::max(global_let_ms, 1e-6)) << "x\n";
                std::cout << std::setprecision(4);
                std::cout << "  Mean Relative Force Err:   " << std::setw(8) << (global_sum_err / total_n) * 100.0 << "%\n";
                std::cout << "  Max  Relative Force Err:   " << std::setw(8) << global_max_err * 100.0 << "%\n";
                std::cout << "--------------------------------------------------------\n\n";
            }

            // Restore distributed acceleration state
            ps.ax = std::move(dist_ax);
            ps.ay = std::move(dist_ay);
            ps.az = std::move(dist_az);
        } else if (mpi.is_root()) {
            std::cout << "--- Running Baseline Comparison (Step 0) ---\n";
            auto t0 = std::chrono::high_resolution_clock::now();
            backend->direct_compute_forces(ps, g_val, eps_sq);
            auto t1 = std::chrono::high_resolution_clock::now();
            double t_direct = std::chrono::duration<double, std::milli>(t1 - t0).count();

            std::vector<float> d_ax = ps.ax;
            std::vector<float> d_ay = ps.ay;
            std::vector<float> d_az = ps.az;

            auto t2 = std::chrono::high_resolution_clock::now();
            backend->compute_forces(ps, theta, g_val, eps_sq);
            auto t3 = std::chrono::high_resolution_clock::now();
            double t_bh = std::chrono::duration<double, std::milli>(t3 - t2).count();

            double max_rel_err = 0.0;
            double sum_rel_err = 0.0;
            for (int i = 0; i < n; i++) {
                double d_mag = std::sqrt(d_ax[i]*d_ax[i] + d_ay[i]*d_ay[i] + d_az[i]*d_az[i]);
                double diff_x = ps.ax[i] - d_ax[i];
                double diff_y = ps.ay[i] - d_ay[i];
                double diff_z = ps.az[i] - d_az[i];
                double diff_mag = std::sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);

                double rel_err = (d_mag > 1e-8) ? (diff_mag / d_mag) : diff_mag;
                if (rel_err > max_rel_err) max_rel_err = rel_err;
                sum_rel_err += rel_err;
            }

            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  Direct O(N^2) Time:      " << std::setw(8) << t_direct << " ms\n";
            std::cout << "  Barnes-Hut Time:         " << std::setw(8) << t_bh << " ms\n";
            std::cout << "  Speedup:                 " << std::setw(8) << (t_direct / std::max(t_bh, 1e-6)) << "x\n";
            std::cout << std::setprecision(4);
            std::cout << "  Mean Relative Force Err: " << std::setw(8) << (sum_rel_err / n) * 100.0 << "%\n";
            std::cout << "  Max  Relative Force Err: " << std::setw(8) << max_rel_err * 100.0 << "%\n";
            std::cout << "--------------------------------------------\n\n";
        }
    }

    if (n <= 4096 && mpi.is_root() && !mpi.enabled) {
        double ke0, pe0;
        ps.compute_energy(g_val, eps_sq, ke0, pe0);
        std::cout << std::scientific << std::setprecision(4);
        std::cout << "Initial Energy -> KE: " << ke0 << " | PE: " << pe0 << " | Total: " << (ke0 + pe0) << "\n\n";
    }

    if (!dump_dir.empty() && mpi.is_root()) {
        mkdir(dump_dir.c_str(), 0777);
    }

    auto start_sim = std::chrono::high_resolution_clock::now();
    float sim_time = 0.0f;

    for (int s = 0; s < steps; s++) {
        if (!dump_dir.empty() && (s % dump_interval == 0) && mpi.is_root()) {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/step_%05d.bin", dump_dir.c_str(), s);
            write_snapshot(path, ps, s, sim_time, dt);
        }

        if (mpi.enabled) {
            // Distributed Locally Essential Tree (LET) cycle
            BoundingBox g_box = mpi.global_bounding_box(ps);
            mpi.migrate_particles(ps, g_box);

            if (use_direct) {
                backend->direct_compute_forces(ps, g_val, eps_sq);
            } else {
                backend->compute_forces(ps, theta, g_val, eps_sq);
            }

            // Extract and exchange coarse subtree multipoles
            std::vector<RemoteMultipole> local_coarse;
            backend->extract_coarse_nodes(2, mpi.rank, local_coarse);
            auto remote_multipoles = mpi.exchange_multipoles(local_coarse);

            // Accumulate gravitational forces from coarse remote nodes
#if defined(ASTRO_ENABLE_OPENMP) || defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (size_t i = 0; i < ps.count; i++) {
                const float xi = ps.x[i];
                const float yi = ps.y[i];
                const float zi = ps.z[i];

                float d_ax = 0.0f;
                float d_ay = 0.0f;
                float d_az = 0.0f;

                for (const auto& rm : remote_multipoles) {
                    if (rm.rank == mpi.rank || rm.mass <= 0.0f) continue;
                    const float dx = rm.com_x - xi;
                    const float dy = rm.com_y - yi;
                    const float dz = rm.com_z - zi;
                    const float dist_sq = dx * dx + dy * dy + dz * dz + eps_sq;

                    const float inv_dist = 1.0f / std::sqrt(dist_sq);
                    const float inv_cube = inv_dist * inv_dist * inv_dist;
                    const float scale = g_val * rm.mass * inv_cube;

                    d_ax += scale * dx;
                    d_ay += scale * dy;
                    d_az += scale * dz;
                }

                ps.ax[i] += d_ax;
                ps.ay[i] += d_ay;
                ps.az[i] += d_az;
            }
        } else {
            if (use_direct) {
                backend->direct_compute_forces(ps, g_val, eps_sq);
            } else {
                backend->compute_forces(ps, theta, g_val, eps_sq);
            }
        }

        ps.integrate_symplectic(dt);
        sim_time += dt;

        if (bench_mode && mpi.is_root() && (s % 10 == 0 || s == steps - 1)) {
            std::cout << "[Step " << std::setw(4) << s << "/" << steps << "] sim_time=" << sim_time << "\n";
        }
    }

    auto end_sim = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end_sim - start_sim).count();

    if (mpi.is_root()) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\nSimulation completed in " << total_ms << " ms ("
                  << (total_ms / steps) << " ms/step)\n";

        if (n <= 4096 && !mpi.enabled) {
            double ke_final, pe_final;
            ps.compute_energy(g_val, eps_sq, ke_final, pe_final);
            std::cout << std::scientific << std::setprecision(4);
            std::cout << "Final   Energy -> KE: " << ke_final << " | PE: " << pe_final << " | Total: " << (ke_final + pe_final) << "\n";
        }
    }

    mpi.finalize();
    return 0;
}
