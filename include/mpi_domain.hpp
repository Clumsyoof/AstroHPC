#ifndef MPI_DOMAIN_HPP
#define MPI_DOMAIN_HPP

#include "particles.hpp"
#include <vector>
#include <cstdint>

#ifdef ASTRO_ENABLE_MPI
#include <mpi.h>
#endif

namespace astro {

struct RemoteMultipole {
    float com_x, com_y, com_z;
    float mass;
    float half_size;
    float cx, cy, cz;
    int rank;
};

struct GlobalParticle {
    float x, y, z, m;
};

class MpiContext {
public:
    int rank = 0;
    int size = 1;
    bool enabled = false;

    bool is_root() const { return rank == 0; }

    static MpiContext init(int* argc, char*** argv);
    void finalize();

    // Synchronizes the 3D bounding box globally across all MPI ranks
    BoundingBox global_bounding_box(const ParticleSystem& ps) const;

    // Partitions particles among ranks using Morton space-filling curve ranges
    void partition_particles(ParticleSystem& ps) const;

    // Migrates particles across domain boundaries to their target rank
    void migrate_particles(ParticleSystem& ps, const BoundingBox& global_bbox) const;

    // Exchanges coarse subtrees/multipoles for distributed Locally Essential Tree (LET) evaluation
    std::vector<RemoteMultipole> exchange_multipoles(const std::vector<RemoteMultipole>& local_nodes) const;

    // Gathers all particles across ranks for distributed verification (--compare)
    std::vector<GlobalParticle> gather_all_particles(const ParticleSystem& ps, std::vector<int>& out_displs) const;
};

} // namespace astro

#endif // MPI_DOMAIN_HPP
