#ifndef PARTICLES_HPP
#define PARTICLES_HPP

#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>
#include "morton.hpp"

namespace astro {

class ParticleSystem {
public:
    size_t count = 0;

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

    // Bounding box & Morton space-filling curve sorting
    BoundingBox compute_bounding_box() const;
    void compute_morton_keys();
    void sort_by_morton();

    // Physics
    void reset_accelerations();
    void integrate_symplectic(float dt);
    void integrate_symplectic_reverse_pos(float dt);
    void integrate_symplectic_reverse_vel(float dt);
    void compute_energy(float G, float eps_sq, double& kinetic, double& potential) const;

    // Presets & Loaders
    void init_three_body();
    void init_disk(size_t n, float radius, float central_mass, float total_disk_mass);
    bool load_csv(const std::string& filepath);
};

} // namespace astro

#endif // PARTICLES_HPP
