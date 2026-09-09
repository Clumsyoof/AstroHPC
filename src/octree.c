#include "octree.h"
#include <math.h>
#include <float.h>
#include <stdio.h>

void octree_pool_reset(OctreePool *pool) {
    pool->node_count = 0;
}

int octree_alloc_node(OctreePool *pool, float cx, float cy, float cz, float half_size) {
    if (pool->node_count >= MAX_OCTREE_NODES) {
        static int warned = 0;
        if (!warned) {
            fprintf(stderr, "\n[astrohpc] Warning: Octree node pool exhausted (MAX_OCTREE_NODES = %d).\n"
                            "           Subsequent particles will be dropped from spatial tree!\n",
                    MAX_OCTREE_NODES);
            warned = 1;
        }
        return -1;
    }
    int idx = pool->node_count++;
    OctNode *node = &pool->nodes[idx];
    node->cx = cx;
    node->cy = cy;
    node->cz = cz;
    node->half_size = half_size;
    node->com_x = 0.0f;
    node->com_y = 0.0f;
    node->com_z = 0.0f;
    node->mass = 0.0f;
    node->body_idx = -1;
    for (int i = 0; i < 8; i++) {
        node->children[i] = -1;
    }
    return idx;
}

static inline int get_octant(const OctNode *node, float px, float py, float pz) {
    int octant = 0;
    if (px >= node->cx) octant |= 1;
    if (py >= node->cy) octant |= 2;
    if (pz >= node->cz) octant |= 4;
    return octant;
}

static inline void get_child_center(const OctNode *node, int octant, float *cx, float *cy, float *cz, float *half_size) {
    *half_size = node->half_size * 0.5f;
    *cx = node->cx + ((octant & 1) ? *half_size : -*half_size);
    *cy = node->cy + ((octant & 2) ? *half_size : -*half_size);
    *cz = node->cz + ((octant & 4) ? *half_size : -*half_size);
}

static int insert_particle(OctreePool *pool, const Particles *sys, int node_idx, int body_idx, int depth) {
    const float px = sys->x[body_idx];
    const float py = sys->y[body_idx];
    const float pz = sys->z[body_idx];
    const float pm = sys->m[body_idx];

    OctNode *node = &pool->nodes[node_idx];

    // On-the-fly Center of Mass update
    const float m_old = node->mass;
    const float m_new = m_old + pm;
    if (m_new > 0.0f) {
        node->com_x = (m_old * node->com_x + pm * px) / m_new;
        node->com_y = (m_old * node->com_y + pm * py) / m_new;
        node->com_z = (m_old * node->com_z + pm * pz) / m_new;
    }
    node->mass = m_new;

    // Case 1: Empty node -> become a leaf containing this particle
    if (m_old == 0.0f && node->body_idx == -1) {
        node->body_idx = body_idx;
        return 0;
    }

    // Maximum depth check to avoid infinite recursion on identical positions
    if (depth >= MAX_OCTREE_DEPTH) {
        return 0;
    }

    // Case 2: Node is a leaf with an existing particle -> split into internal node
    if (node->body_idx >= 0) {
        const int existing_body = node->body_idx;
        node->body_idx = -1; // Convert to internal node

        // Route existing particle into child
        const float ex_px = sys->x[existing_body];
        const float ex_py = sys->y[existing_body];
        const float ex_pz = sys->z[existing_body];
        const int oct_ex = get_octant(node, ex_px, ex_py, ex_pz);

        if (node->children[oct_ex] == -1) {
            float child_cx, child_cy, child_cz, child_hs;
            get_child_center(node, oct_ex, &child_cx, &child_cy, &child_cz, &child_hs);
            int child_idx = octree_alloc_node(pool, child_cx, child_cy, child_cz, child_hs);
            if (child_idx == -1) {
                // Restore existing particle so we don't lose both
                node->body_idx = existing_body;
                return -1;
            }
            node->children[oct_ex] = child_idx;
        }
        if (insert_particle(pool, sys, node->children[oct_ex], existing_body, depth + 1) != 0) {
            return -1;
        }

        // Refresh pointer after recursive insertion
        node = &pool->nodes[node_idx];
    }

    // Case 3: Node is internal -> route new particle into corresponding child
    const int oct_new = get_octant(node, px, py, pz);
    if (node->children[oct_new] == -1) {
        float child_cx, child_cy, child_cz, child_hs;
        get_child_center(node, oct_new, &child_cx, &child_cy, &child_cz, &child_hs);
        int child_idx = octree_alloc_node(pool, child_cx, child_cy, child_cz, child_hs);
        if (child_idx == -1) {
            return -1;
        }
        node->children[oct_new] = child_idx;
    }
    return insert_particle(pool, sys, node->children[oct_new], body_idx, depth + 1);
}

int octree_build(OctreePool *pool, const Particles *sys, int n) {
    octree_pool_reset(pool);

    if (n <= 0) return -1;

    // Determine spatial bounding box
    float min_x = FLT_MAX, min_y = FLT_MAX, min_z = FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX, max_z = -FLT_MAX;

    for (int i = 0; i < n; i++) {
        if (sys->x[i] < min_x) min_x = sys->x[i];
        if (sys->x[i] > max_x) max_x = sys->x[i];
        if (sys->y[i] < min_y) min_y = sys->y[i];
        if (sys->y[i] > max_y) max_y = sys->y[i];
        if (sys->z[i] < min_z) min_z = sys->z[i];
        if (sys->z[i] > max_z) max_z = sys->z[i];
    }

    const float cx = (min_x + max_x) * 0.5f;
    const float cy = (min_y + max_y) * 0.5f;
    const float cz = (min_z + max_z) * 0.5f;

    float max_dim = max_x - min_x;
    if (max_y - min_y > max_dim) max_dim = max_y - min_y;
    if (max_z - min_z > max_dim) max_dim = max_z - min_z;

    // Half size with margin to guarantee all bodies fall strictly within root cube
    const float half_size = (max_dim * 0.5f) + 1.0f;

    int root = octree_alloc_node(pool, cx, cy, cz, half_size);
    if (root == -1) return -1;

    int dropped = 0;
    for (int i = 0; i < n; i++) {
        if (insert_particle(pool, sys, root, i, 0) != 0) {
            dropped++;
        }
    }

    if (dropped > 0) {
        static int warned_drop = 0;
        if (!warned_drop) {
            fprintf(stderr, "[astrohpc] Warning: %d / %d bodies dropped from octree due to node pool exhaustion!\n",
                    dropped, n);
            warned_drop = 1;
        }
    }

    return root;
}

void octree_compute_forces(const OctreePool *pool, Particles *sys, int n, float theta, float G, float eps_sq) {
    if (pool->node_count <= 0 || n <= 0) return;

    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < n; i++) {
        const float xi = sys->x[i];
        const float yi = sys->y[i];
        const float zi = sys->z[i];

        float ax = 0.0f;
        float ay = 0.0f;
        float az = 0.0f;

        // Fixed-size traversal stack per thread
        int stack[256];
        int stack_top = 0;
        stack[stack_top++] = 0; // Root node index

        while (stack_top > 0) {
            const int node_idx = stack[--stack_top];
            const OctNode *node = &pool->nodes[node_idx];

            if (node->mass <= 0.0f) continue;

            // Direct leaf interaction
            if (node->body_idx >= 0) {
                if (node->body_idx == i) continue; // Ignore self-gravity

                const float dx = node->com_x - xi;
                const float dy = node->com_y - yi;
                const float dz = node->com_z - zi;
                const float dist_sq = dx*dx + dy*dy + dz*dz + eps_sq;

                const float inv_dist = 1.0f / sqrtf(dist_sq);
                const float inv_cube = inv_dist * inv_dist * inv_dist;
                const float s = G * node->mass * inv_cube;

                ax += s * dx;
                ay += s * dy;
                az += s * dz;
                continue;
            }

            // Internal node: Check Multipole Acceptance Criterion (MAC)
            const float dx = node->com_x - xi;
            const float dy = node->com_y - yi;
            const float dz = node->com_z - zi;
            const float dist_sq = dx*dx + dy*dy + dz*dz;
            const float dist = sqrtf(dist_sq);

            // MAC: (2 * half_size) / dist < theta
            if ((2.0f * node->half_size) < theta * dist) {
                const float dist_sq_soft = dist_sq + eps_sq;
                const float inv_dist = 1.0f / sqrtf(dist_sq_soft);
                const float inv_cube = inv_dist * inv_dist * inv_dist;
                const float s = G * node->mass * inv_cube;

                ax += s * dx;
                ay += s * dy;
                az += s * dz;
            } else {
                // Criterion fails: traverse valid children
                for (int c = 0; c < 8; c++) {
                    const int child_idx = node->children[c];
                    if (child_idx != -1 && pool->nodes[child_idx].mass > 0.0f) {
                        stack[stack_top++] = child_idx;
                    }
                }
            }
        }

        sys->ax[i] = ax;
        sys->ay[i] = ay;
        sys->az[i] = az;
    }
}

void direct_compute_forces(Particles *sys, int n, float G, float eps_sq) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float ax = 0.0f;
        float ay = 0.0f;
        float az = 0.0f;

        const float xi = sys->x[i];
        const float yi = sys->y[i];
        const float zi = sys->z[i];

        #pragma GCC ivdep
        for (int j = 0; j < n; j++) {
            if (i == j) continue;

            const float dx = sys->x[j] - xi;
            const float dy = sys->y[j] - yi;
            const float dz = sys->z[j] - zi;

            const float dist_sq = dx*dx + dy*dy + dz*dz + eps_sq;
            const float inv_dist = 1.0f / sqrtf(dist_sq);
            const float inv_cube = inv_dist * inv_dist * inv_dist;

            const float s = G * sys->m[j] * inv_cube;

            ax += s * dx;
            ay += s * dy;
            az += s * dz;
        }

        sys->ax[i] = ax;
        sys->ay[i] = ay;
        sys->az[i] = az;
    }
}
