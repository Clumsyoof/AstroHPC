#include "cpu_backend.hpp"
#include <cmath>
#include <algorithm>

namespace astro {

#ifndef MAX_OCTREE_DEPTH
#define MAX_OCTREE_DEPTH 32
#endif

void CpuBackend::reset_pool() {
    nodes.clear();
}

int CpuBackend::alloc_node(float cx, float cy, float cz, float half_size) {
    int idx = static_cast<int>(nodes.size());
    CpuOctNode node;
    node.cx = cx;
    node.cy = cy;
    node.cz = cz;
    node.half_size = half_size;
    node.com_x = 0.0f;
    node.com_y = 0.0f;
    node.com_z = 0.0f;
    node.mass = 0.0f;
    node.body_idx = -1;
    for (int i = 0; i < 8; i++) {
        node.children[i] = -1;
    }
    nodes.push_back(node);
    return idx;
}

static inline int get_octant(float cx, float cy, float cz, float px, float py, float pz) {
    int octant = 0;
    if (px >= cx) octant |= 1;
    if (py >= cy) octant |= 2;
    if (pz >= cz) octant |= 4;
    return octant;
}

static inline void get_child_center(float cx, float cy, float cz, float half_size, int octant,
                                    float& out_cx, float& out_cy, float& out_cz, float& out_hs) {
    out_hs = half_size * 0.5f;
    out_cx = cx + ((octant & 1) ? out_hs : -out_hs);
    out_cy = cy + ((octant & 2) ? out_hs : -out_hs);
    out_cz = cz + ((octant & 4) ? out_hs : -out_hs);
}

int CpuBackend::insert_particle(const ParticleSystem& ps, int node_idx, int body_idx, int depth) {
    const float px = ps.x[body_idx];
    const float py = ps.y[body_idx];
    const float pz = ps.z[body_idx];
    const float pm = ps.m[body_idx];

    // On-the-fly Center of Mass update
    const float m_old = nodes[node_idx].mass;
    const float m_new = m_old + pm;
    if (m_new > 0.0f) {
        nodes[node_idx].com_x = (m_old * nodes[node_idx].com_x + pm * px) / m_new;
        nodes[node_idx].com_y = (m_old * nodes[node_idx].com_y + pm * py) / m_new;
        nodes[node_idx].com_z = (m_old * nodes[node_idx].com_z + pm * pz) / m_new;
    }
    nodes[node_idx].mass = m_new;

    // Case 1: Empty leaf
    if (m_old == 0.0f && nodes[node_idx].body_idx == -1) {
        nodes[node_idx].body_idx = body_idx;
        return 0;
    }

    // Depth limit
    if (depth >= MAX_OCTREE_DEPTH) {
        return 0;
    }

    // Case 2: Leaf contains an existing particle -> split into internal node
    if (nodes[node_idx].body_idx >= 0) {
        const int existing_body = nodes[node_idx].body_idx;
        nodes[node_idx].body_idx = -1; // Internal node

        const float ex_px = ps.x[existing_body];
        const float ex_py = ps.y[existing_body];
        const float ex_pz = ps.z[existing_body];
        const int oct_ex = get_octant(nodes[node_idx].cx, nodes[node_idx].cy, nodes[node_idx].cz,
                                      ex_px, ex_py, ex_pz);

        if (nodes[node_idx].children[oct_ex] == -1) {
            float child_cx, child_cy, child_cz, child_hs;
            get_child_center(nodes[node_idx].cx, nodes[node_idx].cy, nodes[node_idx].cz,
                             nodes[node_idx].half_size, oct_ex,
                             child_cx, child_cy, child_cz, child_hs);
            int child_idx = alloc_node(child_cx, child_cy, child_cz, child_hs);
            nodes[node_idx].children[oct_ex] = child_idx;
        }

        insert_particle(ps, nodes[node_idx].children[oct_ex], existing_body, depth + 1);
    }

    // Insert new particle into corresponding octant
    const int oct_new = get_octant(nodes[node_idx].cx, nodes[node_idx].cy, nodes[node_idx].cz,
                                   px, py, pz);

    if (nodes[node_idx].children[oct_new] == -1) {
        float child_cx, child_cy, child_cz, child_hs;
        get_child_center(nodes[node_idx].cx, nodes[node_idx].cy, nodes[node_idx].cz,
                         nodes[node_idx].half_size, oct_new,
                         child_cx, child_cy, child_cz, child_hs);
        int child_idx = alloc_node(child_cx, child_cy, child_cz, child_hs);
        nodes[node_idx].children[oct_new] = child_idx;
    }

    return insert_particle(ps, nodes[node_idx].children[oct_new], body_idx, depth + 1);
}

int CpuBackend::build_tree(const ParticleSystem& ps) {
    reset_pool();
    if (ps.count == 0) return -1;

    nodes.reserve(ps.count * 4);

    BoundingBox bbox = ps.compute_bounding_box();
    const float cx = (bbox.min_x + bbox.max_x) * 0.5f;
    const float cy = (bbox.min_y + bbox.max_y) * 0.5f;
    const float cz = (bbox.min_z + bbox.max_z) * 0.5f;
    const float half_size = (bbox.max_extent() * 0.5f) + 1.0f;

    int root = alloc_node(cx, cy, cz, half_size);

    for (size_t i = 0; i < ps.count; i++) {
        insert_particle(ps, root, static_cast<int>(i), 0);
    }

    return root;
}

void CpuBackend::compute_forces(ParticleSystem& ps, float theta, float G, float eps_sq) {
    if (ps.count == 0) return;

    build_tree(ps);
    if (nodes.empty()) return;

    for (size_t i = 0; i < ps.count; i++) {
        const float xi = ps.x[i];
        const float yi = ps.y[i];
        const float zi = ps.z[i];

        float ax = 0.0f;
        float ay = 0.0f;
        float az = 0.0f;

        int stack[256];
        int stack_top = 0;
        stack[stack_top++] = 0; // Root node

        while (stack_top > 0) {
            const int node_idx = stack[--stack_top];
            const CpuOctNode& node = nodes[node_idx];

            if (node.mass <= 0.0f) continue;

            // Direct leaf interaction
            if (node.body_idx >= 0) {
                if (node.body_idx == static_cast<int>(i)) continue;

                const float dx = node.com_x - xi;
                const float dy = node.com_y - yi;
                const float dz = node.com_z - zi;
                const float dist_sq = dx * dx + dy * dy + dz * dz + eps_sq;

                const float inv_dist = 1.0f / std::sqrt(dist_sq);
                const float inv_cube = inv_dist * inv_dist * inv_dist;
                const float s = G * node.mass * inv_cube;

                ax += s * dx;
                ay += s * dy;
                az += s * dz;
                continue;
            }

            // Internal node: MAC evaluation
            const float dx = node.com_x - xi;
            const float dy = node.com_y - yi;
            const float dz = node.com_z - zi;
            const float dist_sq = dx * dx + dy * dy + dz * dz;
            const float dist = std::sqrt(dist_sq);

            if ((2.0f * node.half_size) < theta * dist) {
                const float dist_sq_soft = dist_sq + eps_sq;
                const float inv_dist = 1.0f / std::sqrt(dist_sq_soft);
                const float inv_cube = inv_dist * inv_dist * inv_dist;
                const float s = G * node.mass * inv_cube;

                ax += s * dx;
                ay += s * dy;
                az += s * dz;
            } else {
                for (int c = 0; c < 8; c++) {
                    const int child_idx = node.children[c];
                    if (child_idx != -1 && nodes[child_idx].mass > 0.0f) {
                        stack[stack_top++] = child_idx;
                    }
                }
            }
        }

        ps.ax[i] = ax;
        ps.ay[i] = ay;
        ps.az[i] = az;
    }
}

void CpuBackend::direct_compute_forces(ParticleSystem& ps, float G, float eps_sq) {
    for (size_t i = 0; i < ps.count; i++) {
        const float xi = ps.x[i];
        const float yi = ps.y[i];
        const float zi = ps.z[i];

        float ax = 0.0f;
        float ay = 0.0f;
        float az = 0.0f;

        for (size_t j = 0; j < ps.count; j++) {
            if (i == j) continue;
            const float dx = ps.x[j] - xi;
            const float dy = ps.y[j] - yi;
            const float dz = ps.z[j] - zi;
            const float dist_sq = dx * dx + dy * dy + dz * dz + eps_sq;

            const float inv_dist = 1.0f / std::sqrt(dist_sq);
            const float inv_cube = inv_dist * inv_dist * inv_dist;
            const float s = G * ps.m[j] * inv_cube;

            ax += s * dx;
            ay += s * dy;
            az += s * dz;
        }

        ps.ax[i] = ax;
        ps.ay[i] = ay;
        ps.az[i] = az;
    }
}

} // namespace astro
