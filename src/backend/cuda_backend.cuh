#ifndef CUDA_BACKEND_CUH
#define CUDA_BACKEND_CUH

#include "backend.hpp"

#ifdef ASTRO_ENABLE_CUDA

namespace astro {

class CudaBackend : public IComputeBackend {
public:
    CudaBackend();
    ~CudaBackend() override;

    const char* name() const override { return "CUDA-Tile"; }

    void compute_forces(ParticleSystem& ps, float theta, float G, float eps_sq) override;
    void direct_compute_forces(ParticleSystem& ps, float G, float eps_sq) override;

private:
    void ensure_capacity(size_t n);

    void* d_pos_mass = nullptr;
    void* d_acc = nullptr;
    size_t capacity = 0;

    std::vector<float> h_pos_mass_flat;
    std::vector<float> h_acc_flat;
};

} // namespace astro

#endif // ASTRO_ENABLE_CUDA

#endif // CUDA_BACKEND_CUH
