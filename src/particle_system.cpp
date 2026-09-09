#include "particles.hpp"
#include "config.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>

#if defined(ASTRO_ENABLE_OPENMP) || defined(_OPENMP)
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace astro {

float ParticleSystem::default_g() const {
    switch (units) {
        case DatasetUnits::SolarSystem: return G_SOLAR_SYSTEM;
        case DatasetUnits::Galactic:    return G_IRL_ASTRO;
        default:                        return DEFAULT_G;
    }
}

float ParticleSystem::default_eps_sq() const {
    switch (units) {
        case DatasetUnits::SolarSystem: return DEFAULT_EPS_SQ_SOLAR;
        case DatasetUnits::Galactic:    return DEFAULT_EPSILON_SQ;
        default:                        return DEFAULT_EPSILON_SQ;
    }
}

float ParticleSystem::default_dt() const {
    switch (units) {
        case DatasetUnits::SolarSystem: return DEFAULT_DT_SOLAR;
        case DatasetUnits::Galactic:    return DEFAULT_DT;
        default:                        return DEFAULT_DT;
    }
}

void ParticleSystem::resize(size_t n) {
    count = n;
    names.resize(n);
    x.resize(n, 0.0f);
    y.resize(n, 0.0f);
    z.resize(n, 0.0f);
    vx.resize(n, 0.0f);
    vy.resize(n, 0.0f);
    vz.resize(n, 0.0f);
    ax.resize(n, 0.0f);
    ay.resize(n, 0.0f);
    az.resize(n, 0.0f);
    m.resize(n, 1.0f);
    morton.resize(n, 0ULL);
}

void ParticleSystem::clear() {
    count = 0;
    units = DatasetUnits::Dimensionless;
    names.clear();
    x.clear(); y.clear(); z.clear();
    vx.clear(); vy.clear(); vz.clear();
    ax.clear(); ay.clear(); az.clear();
    m.clear(); morton.clear();
}

void ParticleSystem::swap_particles(size_t i, size_t j) {
    if (i == j) return;
    if (i < names.size() && j < names.size()) {
        std::swap(names[i], names[j]);
    }
    std::swap(x[i], x[j]);
    std::swap(y[i], y[j]);
    std::swap(z[i], z[j]);
    std::swap(vx[i], vx[j]);
    std::swap(vy[i], vy[j]);
    std::swap(vz[i], vz[j]);
    std::swap(ax[i], ax[j]);
    std::swap(ay[i], ay[j]);
    std::swap(az[i], az[j]);
    std::swap(m[i], m[j]);
    std::swap(morton[i], morton[j]);
}

BoundingBox ParticleSystem::compute_bounding_box() const {
    BoundingBox box{0, 0, 0, 0, 0, 0};
    if (count == 0) return box;

    box.min_x = box.max_x = x[0];
    box.min_y = box.max_y = y[0];
    box.min_z = box.max_z = z[0];

    for (size_t i = 1; i < count; i++) {
        if (x[i] < box.min_x) box.min_x = x[i];
        if (x[i] > box.max_x) box.max_x = x[i];
        if (y[i] < box.min_y) box.min_y = y[i];
        if (y[i] > box.max_y) box.max_y = y[i];
        if (z[i] < box.min_z) box.min_z = z[i];
        if (z[i] > box.max_z) box.max_z = z[i];
    }
    return box;
}

void ParticleSystem::compute_morton_keys() {
    BoundingBox box = compute_bounding_box();
#if defined(ASTRO_ENABLE_OPENMP) || defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < count; i++) {
        morton[i] = morton_encode_3d(x[i], y[i], z[i], box);
    }
}

void ParticleSystem::sort_by_morton() {
    if (count <= 1) return;

    compute_morton_keys();

    // Permutation vector
    std::vector<size_t> p(count);
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), [this](size_t a, size_t b) {
        return morton[a] < morton[b];
    });

    // Reorder channels according to permutation
    std::vector<float> tmp_x(count), tmp_y(count), tmp_z(count);
    std::vector<float> tmp_vx(count), tmp_vy(count), tmp_vz(count);
    std::vector<float> tmp_ax(count), tmp_ay(count), tmp_az(count);
    std::vector<float> tmp_m(count);
    std::vector<uint64_t> tmp_morton(count);

    for (size_t i = 0; i < count; i++) {
        size_t src = p[i];
        tmp_x[i] = x[src];
        tmp_y[i] = y[src];
        tmp_z[i] = z[src];
        tmp_vx[i] = vx[src];
        tmp_vy[i] = vy[src];
        tmp_vz[i] = vz[src];
        tmp_ax[i] = ax[src];
        tmp_ay[i] = ay[src];
        tmp_az[i] = az[src];
        tmp_m[i] = m[src];
        tmp_morton[i] = morton[src];
    }

    std::vector<std::string> tmp_names(count);
    for (size_t i = 0; i < count; i++) {
        size_t src = p[i];
        if (src < names.size()) tmp_names[i] = names[src];
    }
    if (!names.empty()) names = std::move(tmp_names);

    x = std::move(tmp_x);
    y = std::move(tmp_y);
    z = std::move(tmp_z);
    vx = std::move(tmp_vx);
    vy = std::move(tmp_vy);
    vz = std::move(tmp_vz);
    ax = std::move(tmp_ax);
    ay = std::move(tmp_ay);
    az = std::move(tmp_az);
    m = std::move(tmp_m);
    morton = std::move(tmp_morton);
}

void ParticleSystem::reset_accelerations() {
    std::fill(ax.begin(), ax.end(), 0.0f);
    std::fill(ay.begin(), ay.end(), 0.0f);
    std::fill(az.begin(), az.end(), 0.0f);
}

void ParticleSystem::integrate_symplectic(float dt) {
#if defined(ASTRO_ENABLE_OPENMP) || defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < count; i++) {
        vx[i] += ax[i] * dt;
        vy[i] += ay[i] * dt;
        vz[i] += az[i] * dt;

        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;
        z[i] += vz[i] * dt;
    }
}

void ParticleSystem::integrate_symplectic_reverse_pos(float dt) {
#if defined(ASTRO_ENABLE_OPENMP) || defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < count; i++) {
        x[i] -= vx[i] * dt;
        y[i] -= vy[i] * dt;
        z[i] -= vz[i] * dt;
    }
}

void ParticleSystem::integrate_symplectic_reverse_vel(float dt) {
#if defined(ASTRO_ENABLE_OPENMP) || defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < count; i++) {
        vx[i] -= ax[i] * dt;
        vy[i] -= ay[i] * dt;
        vz[i] -= az[i] * dt;
    }
}

void ParticleSystem::kick(float dt_half) {
#if defined(ASTRO_ENABLE_OPENMP) || defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < count; i++) {
        vx[i] += ax[i] * dt_half;
        vy[i] += ay[i] * dt_half;
        vz[i] += az[i] * dt_half;
    }
}

void ParticleSystem::drift(float dt) {
#if defined(ASTRO_ENABLE_OPENMP) || defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < count; i++) {
        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;
        z[i] += vz[i] * dt;
    }
}

void ParticleSystem::compute_energy(float G, float eps_sq, double& kinetic, double& potential) const {
    double total_ke = 0.0;
    double total_pe = 0.0;

#if defined(ASTRO_ENABLE_OPENMP) || defined(_OPENMP)
#pragma omp parallel for reduction(+:total_ke) schedule(static)
#endif
    for (size_t i = 0; i < count; i++) {
        double v_sq = (double)vx[i] * vx[i] + (double)vy[i] * vy[i] + (double)vz[i] * vz[i];
        total_ke += 0.5 * (double)m[i] * v_sq;
    }

#if defined(ASTRO_ENABLE_OPENMP) || defined(_OPENMP)
#pragma omp parallel for reduction(+:total_pe) schedule(guided)
#endif
    for (size_t i = 0; i < count; i++) {
        float xi = x[i];
        float yi = y[i];
        float zi = z[i];
        float mi = m[i];

        for (size_t j = i + 1; j < count; j++) {
            float dx = x[j] - xi;
            float dy = y[j] - yi;
            float dz = z[j] - zi;
            float dist = std::sqrt(dx * dx + dy * dy + dz * dz + eps_sq);
            total_pe -= (double)(G * mi * m[j]) / (double)dist;
        }
    }

    kinetic = total_ke;
    potential = total_pe;
}

void ParticleSystem::compute_momentum(double& px, double& py, double& pz) const {
    double t_px = 0.0;
    double t_py = 0.0;
    double t_pz = 0.0;

#if defined(ASTRO_ENABLE_OPENMP) || defined(_OPENMP)
#pragma omp parallel for reduction(+:t_px, t_py, t_pz) schedule(static)
#endif
    for (size_t i = 0; i < count; i++) {
        t_px += (double)m[i] * (double)vx[i];
        t_py += (double)m[i] * (double)vy[i];
        t_pz += (double)m[i] * (double)vz[i];
    }
    px = t_px;
    py = t_py;
    pz = t_pz;
}

void ParticleSystem::compute_angular_momentum(double& lx, double& ly, double& lz) const {
    double t_lx = 0.0;
    double t_ly = 0.0;
    double t_lz = 0.0;

#if defined(ASTRO_ENABLE_OPENMP) || defined(_OPENMP)
#pragma omp parallel for reduction(+:t_lx, t_ly, t_lz) schedule(static)
#endif
    for (size_t i = 0; i < count; i++) {
        double mi = (double)m[i];
        double xi = (double)x[i];
        double yi = (double)y[i];
        double zi = (double)z[i];
        double vxi = (double)vx[i];
        double vyi = (double)vy[i];
        double vzi = (double)vz[i];

        t_lx += mi * (yi * vzi - zi * vyi);
        t_ly += mi * (zi * vxi - xi * vzi);
        t_lz += mi * (xi * vyi - yi * vxi);
    }
    lx = t_lx;
    ly = t_ly;
    lz = t_lz;
}

void ParticleSystem::init_three_body() {
    resize(3);
    reset_accelerations();

    // Central mass
    x[0] = 0.0f;  y[0] = 0.0f;  z[0] = 0.0f;
    vx[0] = 0.0f; vy[0] = 0.0f; vz[0] = 0.0f;
    m[0] = 1000.0f;

    // Body 1
    x[1] = 5.0f;  y[1] = 0.0f;  z[1] = 0.0f;
    vx[1] = 0.0f; vy[1] = 14.0f; vz[1] = 0.0f;
    m[1] = 1.0f;

    // Body 2
    x[2] = 10.0f; y[2] = 0.0f;  z[2] = 0.0f;
    vx[2] = 0.0f; vy[2] = 10.0f; vz[2] = 0.0f;
    m[2] = 1.0f;
}

// Model Qualification: This disk generator models a mock galactic disk with a central point-mass,
// power-law surface density, Salpeter Initial Mass Function (IMF), and small velocity dispersion.
// It is designed for visual exploration, code benchmarking, and numerical validation, rather than
// an exact self-consistent Jeans-theorem equilibrium.
void ParticleSystem::init_disk(size_t n, float radius, float central_mass, float total_disk_mass) {
    resize(n);
    reset_accelerations();

    if (n == 0) return;

    srand(42);

    // Central supermassive body
    x[0] = 0.0f; y[0] = 0.0f; z[0] = 0.0f;
    vx[0] = 0.0f; vy[0] = 0.0f; vz[0] = 0.0f;
    m[0] = central_mass;

    if (n == 1) return;

    // Salpeter Initial Mass Function (IMF): dN/dm ~ m^(-2.35)
    const float gamma = 1.0f - 2.35f;
    const float m_min_g = std::pow(0.15f, gamma);
    const float m_max_g = std::pow(8.0f, gamma);

    float raw_mass_sum = 0.0f;
    for (size_t i = 1; i < n; i++) {
        float u_m = (float)rand() / (float)RAND_MAX;
        m[i] = std::pow(m_min_g + u_m * (m_max_g - m_min_g), 1.0f / gamma);
        raw_mass_sum += m[i];
    }
    const float mass_scale = total_disk_mass / (raw_mass_sum > 1e-4f ? raw_mass_sum : 1.0f);
    for (size_t i = 1; i < n; i++) {
        m[i] *= mass_scale;
    }

    const float min_r = radius * 0.05f;

    for (size_t i = 1; i < n; i++) {
        float u = (float)rand() / (float)RAND_MAX;
        float r = min_r + (radius - min_r) * std::sqrt(u);

        float theta = 2.0f * (float)M_PI * ((float)rand() / (float)RAND_MAX);
        float cos_t = std::cos(theta);
        float sin_t = std::sin(theta);

        float z_scale = radius * 0.02f;
        float z_offset = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * z_scale;

        x[i] = r * cos_t;
        y[i] = r * sin_t;
        z[i] = z_offset;

        float m_enclosed = central_mass + total_disk_mass * (r / radius);
        float v_circ = std::sqrt((DEFAULT_G * m_enclosed) / r);

        float disp = v_circ * 0.03f;
        float vx_disp = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * disp;
        float vy_disp = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * disp;
        float vz_disp = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * disp;

        vx[i] = -v_circ * sin_t + vx_disp;
        vy[i] =  v_circ * cos_t + vy_disp;
        vz[i] = vz_disp;
    }
}

bool ParticleSystem::load_csv(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) return false;

    clear();
    units = DatasetUnits::Dimensionless;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '\r') continue;

        // Parse comments for unit system clues
        if (line[0] == '#') {
            std::string lower_line = line;
            std::transform(lower_line.begin(), lower_line.end(), lower_line.begin(), ::tolower);
            if (lower_line.find("solar system") != std::string::npos ||
                lower_line.find("39.478") != std::string::npos ||
                lower_line.find("distance in au") != std::string::npos ||
                lower_line.find("horizons") != std::string::npos) {
                units = DatasetUnits::SolarSystem;
            } else if (lower_line.find("gaia") != std::string::npos ||
                       lower_line.find("pleiades") != std::string::npos ||
                       lower_line.find("0.0043009") != std::string::npos ||
                       lower_line.find("pos=pc") != std::string::npos) {
                units = DatasetUnits::Galactic;
            }
            continue;
        }

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() >= 5) {
            try {
                // name, mass, x, y, z, [vx, vy, vz]
                std::string body_name = tokens[0];
                float mass_val = std::stof(tokens[1]);
                float px = std::stof(tokens[2]);
                float py = std::stof(tokens[3]);
                float pz = std::stof(tokens[4]);
                float vel_x = (tokens.size() >= 8) ? std::stof(tokens[5]) : 0.0f;
                float vel_y = (tokens.size() >= 8) ? std::stof(tokens[6]) : 0.0f;
                float vel_z = (tokens.size() >= 8) ? std::stof(tokens[7]) : 0.0f;

                names.push_back(body_name);
                x.push_back(px);
                y.push_back(py);
                z.push_back(pz);
                vx.push_back(vel_x);
                vy.push_back(vel_y);
                vz.push_back(vel_z);
                ax.push_back(0.0f);
                ay.push_back(0.0f);
                az.push_back(0.0f);
                m.push_back(mass_val);
                morton.push_back(0ULL);
                count++;
            } catch (...) {
                // Skip header line or malformed rows
                continue;
            }
        }
    }

    // Secondary fallback based on filepath
    if (units == DatasetUnits::Dimensionless) {
        std::string lower_path = filepath;
        std::transform(lower_path.begin(), lower_path.end(), lower_path.begin(), ::tolower);
        if (lower_path.find("solar") != std::string::npos) {
            units = DatasetUnits::SolarSystem;
        } else if (lower_path.find("gaia") != std::string::npos || lower_path.find("pleiades") != std::string::npos) {
            units = DatasetUnits::Galactic;
        }
    }

    return count > 0;
}

} // namespace astro
