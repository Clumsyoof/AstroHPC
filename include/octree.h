#ifndef OCTREE_H
#define OCTREE_H

#include "config.h"
#include "particles.h"

typedef struct {
    float cx, cy, cz;          // Geometric center of spatial bounding cube
    float half_size;           // Half-width of the cube
    float com_x, com_y, com_z; // Center of mass
    float mass;                // Monopole mass (sum of all contained bodies)
    int body_idx;              // -1 for internal/empty node, >= 0 for leaf particle index
    int children[8];           // Indices into pool array (-1 indicates empty)
} OctNode;

typedef struct {
    OctNode nodes[MAX_OCTREE_NODES];
    int node_count;
} OctreePool;

// Initialize / reset the flat memory arena in O(1)
void octree_pool_reset(OctreePool *pool);

// Allocate a new node in the pool arena
int octree_alloc_node(OctreePool *pool, float cx, float cy, float cz, float half_size);

// Build the spatial octree over particles [0, n) and return the root index
int octree_build(OctreePool *pool, const Particles *sys, int n);

// Compute forces using Barnes-Hut multipole traversal (OpenMP accelerated)
void octree_compute_forces(const OctreePool *pool, Particles *sys, int n, float theta, float G, float eps_sq);

// Direct O(N^2) all-pairs force computation for baseline comparison
void direct_compute_forces(Particles *sys, int n, float G, float eps_sq);

#endif // OCTREE_H
