#ifndef CPU_BACKEND_HPP
#define CPU_BACKEND_HPP

#include "backend.hpp"
#include <vector>

namespace astro {

struct CpuOctNode {
    float cx, cy, cz;
    float half_size;
    float com_x, com_y, com_z;
    float mass;
    int body_idx;
    int children[8];
};

class CpuBackend : public IComputeBackend {
public:
    const char* name() const override { return "CPU-BarnesHut"; }

    void compute_forces(ParticleSystem& ps, float theta, float G, float eps_sq) override;
    void direct_compute_forces(ParticleSystem& ps, float G, float eps_sq) override;

    int get_node_count() const { return static_cast<int>(nodes.size()); }
    const std::vector<CpuOctNode>& get_nodes() const { return nodes; }

private:
    std::vector<CpuOctNode> nodes;

    void reset_pool();
    int alloc_node(float cx, float cy, float cz, float half_size);
    int insert_particle(const ParticleSystem& ps, int node_idx, int body_idx, int depth);
    int build_tree(const ParticleSystem& ps);
};

} // namespace astro

#endif // CPU_BACKEND_HPP
