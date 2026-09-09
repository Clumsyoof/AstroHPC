#include "backend.hpp"
#include "cpu_backend.hpp"

#ifdef ASTRO_ENABLE_CUDA
#include "cuda_backend.cuh"
#endif

namespace astro {

std::unique_ptr<IComputeBackend> create_compute_backend() {
#ifdef ASTRO_ENABLE_CUDA
    return std::make_unique<CudaBackend>();
#else
    return std::make_unique<CpuBackend>();
#endif
}

} // namespace astro
