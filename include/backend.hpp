#ifndef BACKEND_HPP
#define BACKEND_HPP

#include <memory>
#include "particles.hpp"

namespace astro {

class IComputeBackend {
public:
    virtual ~IComputeBackend() = default;

    // Backend identifier (e.g. "CPU-BarnesHut", "CUDA-Tile")
    virtual const char* name() const = 0;

    // Barnes-Hut O(N log N) force evaluation
    virtual void compute_forces(ParticleSystem& ps, float theta, float G, float eps_sq) = 0;

    // Direct O(N^2) all-pairs force evaluation
    virtual void direct_compute_forces(ParticleSystem& ps, float G, float eps_sq) = 0;
};

// Factory to instantiate the active backend (modular switch for CPU/CUDA)
std::unique_ptr<IComputeBackend> create_compute_backend();

} // namespace astro

#endif // BACKEND_HPP
