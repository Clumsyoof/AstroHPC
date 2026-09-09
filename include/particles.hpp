#ifndef PARTICLES_HPP
#define PARTICLES_HPP

#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>
#include "morton.hpp"

namespace astro {

enum class DatasetUnits {
    Dimensionless, // G = 1.0 (standard N-body dimensionless units)
    Galactic,      // G = 0.004300917 pc*(km/s)^2/M_sun (Gaia, star clusters)
    SolarSystem    // G = 39.4784176 AU*(AU/yr)^2/M_sun (NASA JPL Horizons ephemerides)
};

class ParticleSystem {
public:
    size_t count = 0;
    DatasetUnits units = DatasetUnits::Dimensionless;
    std::vector<std::string> names;

    // Structure of Arrays (SoA) contiguous channels
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;

    std::vector<float> vx;
    std::vector<float> vy;
    std::vector<float> vz;

    std::vector<float> ax;
    std::vector<float> ay;
    std::vector<float> az;

    std::vector<float> m;
    std::vector<uint64_t> morton;

    void resize(size_t n);
    void clear();
    void swap_particles(size_t i, size_t j);

    // Default physical constants matching current dataset units
    float default_g() const;
    float default_eps_sq() const;
    float default_dt() const;

    // Bounding box & Morton space-filling curve sorting
    BoundingBox compute_bounding_box() const;
    void compute_morton_keys();
    void sort_by_morton();

    // Physics & Symplectic Integrators
    void reset_accelerations();
    void integrate_symplectic(float dt); // Semi-implicit Euler (1st order symplectic)
    void integrate_symplectic_reverse_pos(float dt);
    void integrate_symplectic_reverse_vel(float dt);

    // Velocity-Verlet / Leapfrog (2nd order symplectic)
    void kick(float dt_half);
    void drift(float dt);

    // Physical Conservation Diagnostics
    void compute_energy(float G, float eps_sq, double& kinetic, double& potential) const;
    void compute_momentum(double& px, double& py, double& pz) const;
    void compute_angular_momentum(double& lx, double& ly, double& lz) const;

    // Presets & Loaders
    void init_three_body();
    void init_disk(size_t n, float radius, float central_mass, float total_disk_mass);
    bool load_csv(const std::string& filepath);
};

} // namespace astro

#endif // PARTICLES_HPP
