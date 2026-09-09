#ifndef CUDA_BACKEND_CUH
#define CUDA_BACKEND_CUH

#include "backend.hpp"

#ifdef ASTRO_ENABLE_CUDA

namespace astro {

class CudaBackend : public IComputeBackend {
public:
    const char* name() const override { return "CUDA-Tile"; }

    void compute_forces(ParticleSystem& ps, float theta, float G, float eps_sq) override;
    void direct_compute_forces(ParticleSystem& ps, float G, float eps_sq) override;
};

} // namespace astro

#endif // ASTRO_ENABLE_CUDA

#endif // CUDA_BACKEND_CUH
