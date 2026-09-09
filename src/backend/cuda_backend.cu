#include "cuda_backend.cuh"

#ifdef ASTRO_ENABLE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>
#include <vector>

namespace astro {

#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[astrohpc CUDA] Error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while (0)

__global__ void nbody_tile_kernel(const float4* __restrict__ pos_mass,
                                  float3* __restrict__ acc,
                                  int n, float G, float eps_sq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float4 my_pos_m = (i < n) ? pos_mass[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 my_acc = make_float3(0.0f, 0.0f, 0.0f);

    __shared__ float4 tile[BLOCK_SIZE];

    int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int idx = t * BLOCK_SIZE + threadIdx.x;
        tile[threadIdx.x] = (idx < n) ? pos_mass[idx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        __syncthreads();

        #pragma unroll 8
        for (int j = 0; j < BLOCK_SIZE; j++) {
            float4 other = tile[j];
            float dx = other.x - my_pos_m.x;
            float dy = other.y - my_pos_m.y;
            float dz = other.z - my_pos_m.z;

            float dist_sq = dx * dx + dy * dy + dz * dz + eps_sq;
            float inv_dist = rsqrtf(dist_sq);
            float inv_cube = inv_dist * inv_dist * inv_dist;
            float s = G * other.w * inv_cube;

            my_acc.x += s * dx;
            my_acc.y += s * dy;
            my_acc.z += s * dz;
        }
        __syncthreads();
    }

    if (i < n) {
        acc[i] = my_acc;
    }
}

CudaBackend::CudaBackend() : d_pos_mass(nullptr), d_acc(nullptr), capacity(0) {}

CudaBackend::~CudaBackend() {
    if (d_pos_mass) {
        cudaFree(d_pos_mass);
        d_pos_mass = nullptr;
    }
    if (d_acc) {
        cudaFree(d_acc);
        d_acc = nullptr;
    }
    capacity = 0;
}

void CudaBackend::ensure_capacity(size_t n) {
    if (n <= capacity) return;

    if (d_pos_mass) {
        cudaFree(d_pos_mass);
        d_pos_mass = nullptr;
    }
    if (d_acc) {
        cudaFree(d_acc);
        d_acc = nullptr;
    }

    size_t new_cap = (n * 3) / 2 + 1024;
    CUDA_CHECK(cudaMalloc(&d_pos_mass, new_cap * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_acc, new_cap * sizeof(float3)));
    capacity = new_cap;
}

void CudaBackend::direct_compute_forces(ParticleSystem& ps, float G, float eps_sq) {
    if (ps.count == 0) return;
    int n = static_cast<int>(ps.count);

    ensure_capacity(ps.count);

    std::vector<float4> h_pos_mass(n);
    for (int i = 0; i < n; i++) {
        h_pos_mass[i] = make_float4(ps.x[i], ps.y[i], ps.z[i], ps.m[i]);
    }

    CUDA_CHECK(cudaMemcpy(d_pos_mass, h_pos_mass.data(), n * sizeof(float4), cudaMemcpyHostToDevice));

    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    nbody_tile_kernel<<<grid_size, BLOCK_SIZE>>>(static_cast<float4*>(d_pos_mass),
                                                 static_cast<float3*>(d_acc),
                                                 n, G, eps_sq);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float3> h_acc(n);
    CUDA_CHECK(cudaMemcpy(h_acc.data(), d_acc, n * sizeof(float3), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        ps.ax[i] = h_acc[i].x;
        ps.ay[i] = h_acc[i].y;
        ps.az[i] = h_acc[i].z;
    }
}

void CudaBackend::compute_forces(ParticleSystem& ps, float theta, float G, float eps_sq) {
    static bool warned_theta = false;
    if (!warned_theta) {
        std::fprintf(stderr,
                     "[astrohpc CUDA] Note: CUDA backend executes shared-memory tiled all-pairs kernel; "
                     "MAC theta=%.2f is bypassed (no GPU octree construction).\n",
                     theta);
        warned_theta = true;
    }
    direct_compute_forces(ps, G, eps_sq);
}

} // namespace astro

#endif // ASTRO_ENABLE_CUDA
