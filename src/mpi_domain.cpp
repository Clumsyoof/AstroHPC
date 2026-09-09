#include "mpi_domain.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>

namespace astro {

MpiContext MpiContext::init(int* argc, char*** argv) {
    MpiContext ctx;
#ifdef ASTRO_ENABLE_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(argc, argv);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.size);
    ctx.enabled = (ctx.size > 1);
#else
    (void)argc;
    (void)argv;
    ctx.rank = 0;
    ctx.size = 1;
    ctx.enabled = false;
#endif
    return ctx;
}

void MpiContext::finalize() {
#ifdef ASTRO_ENABLE_MPI
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
#endif
}

BoundingBox MpiContext::global_bounding_box(const ParticleSystem& ps) const {
    BoundingBox local = ps.compute_bounding_box();
#ifdef ASTRO_ENABLE_MPI
    if (enabled) {
        float l_min[3] = { local.min_x, local.min_y, local.min_z };
        float l_max[3] = { local.max_x, local.max_y, local.max_z };
        float g_min[3], g_max[3];

        MPI_Allreduce(l_min, g_min, 3, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(l_max, g_max, 3, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

        BoundingBox global_box;
        global_box.min_x = g_min[0]; global_box.min_y = g_min[1]; global_box.min_z = g_min[2];
        global_box.max_x = g_max[0]; global_box.max_y = g_max[1]; global_box.max_z = g_max[2];
        return global_box;
    }
#endif
    return local;
}

void MpiContext::partition_particles(ParticleSystem& ps) const {
#ifdef ASTRO_ENABLE_MPI
    if (!enabled || size <= 1) return;

    // Rank 0 sorts particles by Morton keys and scatters equal chunks to other ranks
    int total_n = static_cast<int>(ps.count);
    MPI_Bcast(&total_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int base_count = total_n / size;
    int rem = total_n % size;

    std::vector<int> send_counts(size);
    std::vector<int> displs(size, 0);

    for (int r = 0; r < size; r++) {
        send_counts[r] = base_count + (r < rem ? 1 : 0);
        if (r > 0) {
            displs[r] = displs[r - 1] + send_counts[r - 1];
        }
    }

    int my_count = send_counts[rank];

    auto scatter_channel = [&](std::vector<float>& channel) {
        std::vector<float> local_channel(my_count);
        MPI_Scatterv(rank == 0 ? channel.data() : nullptr,
                     send_counts.data(), displs.data(), MPI_FLOAT,
                     local_channel.data(), my_count, MPI_FLOAT,
                     0, MPI_COMM_WORLD);
        channel = std::move(local_channel);
    };

    if (rank == 0) {
        ps.sort_by_morton();
    }

    scatter_channel(ps.x);
    scatter_channel(ps.y);
    scatter_channel(ps.z);
    scatter_channel(ps.vx);
    scatter_channel(ps.vy);
    scatter_channel(ps.vz);
    scatter_channel(ps.ax);
    scatter_channel(ps.ay);
    scatter_channel(ps.az);
    scatter_channel(ps.m);

    ps.count = static_cast<size_t>(my_count);
    ps.morton.assign(ps.count, 0ULL);
#else
    (void)ps;
#endif
}

void MpiContext::migrate_particles(ParticleSystem& ps, const BoundingBox& global_bbox) const {
#ifdef ASTRO_ENABLE_MPI
    if (!enabled || size <= 1 || ps.count == 0) return;

    // Compute Morton keys using global bounding box
    for (size_t i = 0; i < ps.count; i++) {
        ps.morton[i] = morton_encode_3d(ps.x[i], ps.y[i], ps.z[i], global_bbox);
    }

    const uint64_t stride = UINT64_MAX / static_cast<uint64_t>(size);

    std::vector<int> send_counts(size, 0);
    std::vector<int> target_ranks(ps.count);

    for (size_t i = 0; i < ps.count; i++) {
        int tr = static_cast<int>(ps.morton[i] / stride);
        if (tr >= size) tr = size - 1;
        target_ranks[i] = tr;
        if (tr != rank) {
            send_counts[tr]++;
        }
    }

    std::vector<int> recv_counts(size, 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int total_send = 0;
    int total_recv = 0;
    for (int r = 0; r < size; r++) {
        total_send += send_counts[r];
        total_recv += recv_counts[r];
    }

    if (total_send == 0 && total_recv == 0) return;

    // Structure for serialized particle communication
    struct CommParticle {
        float x, y, z;
        float vx, vy, vz;
        float ax, ay, az;
        float m;
        uint64_t morton;
    };

    std::vector<int> send_displs(size, 0);
    std::vector<int> recv_displs(size, 0);
    for (int r = 1; r < size; r++) {
        send_displs[r] = send_displs[r - 1] + send_counts[r - 1];
        recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
    }

    std::vector<CommParticle> send_buf(total_send);
    std::vector<int> offsets = send_displs;

    std::vector<size_t> keep_indices;
    keep_indices.reserve(ps.count);

    for (size_t i = 0; i < ps.count; i++) {
        int tr = target_ranks[i];
        if (tr == rank) {
            keep_indices.push_back(i);
        } else {
            int pos = offsets[tr]++;
            send_buf[pos] = {
                ps.x[i], ps.y[i], ps.z[i],
                ps.vx[i], ps.vy[i], ps.vz[i],
                ps.ax[i], ps.ay[i], ps.az[i],
                ps.m[i], ps.morton[i]
            };
        }
    }

    std::vector<CommParticle> recv_buf(total_recv);

    std::vector<int> send_bytes(size), recv_bytes(size), send_byte_displs(size), recv_byte_displs(size);
    for (int r = 0; r < size; r++) {
        send_bytes[r] = send_counts[r] * sizeof(CommParticle);
        recv_bytes[r] = recv_counts[r] * sizeof(CommParticle);
        send_byte_displs[r] = send_displs[r] * sizeof(CommParticle);
        recv_byte_displs[r] = recv_displs[r] * sizeof(CommParticle);
    }

    MPI_Alltoallv(send_buf.data(), send_bytes.data(), send_byte_displs.data(), MPI_BYTE,
                  recv_buf.data(), recv_bytes.data(), recv_byte_displs.data(), MPI_BYTE,
                  MPI_COMM_WORLD);

    // Reconstruct local particle system
    size_t new_count = keep_indices.size() + total_recv;
    ParticleSystem new_ps;
    new_ps.resize(new_count);

    for (size_t j = 0; j < keep_indices.size(); j++) {
        size_t src = keep_indices[j];
        new_ps.x[j] = ps.x[src];
        new_ps.y[j] = ps.y[src];
        new_ps.z[j] = ps.z[src];
        new_ps.vx[j] = ps.vx[src];
        new_ps.vy[j] = ps.vy[src];
        new_ps.vz[j] = ps.vz[src];
        new_ps.ax[j] = ps.ax[src];
        new_ps.ay[j] = ps.ay[src];
        new_ps.az[j] = ps.az[src];
        new_ps.m[j] = ps.m[src];
        new_ps.morton[j] = ps.morton[src];
    }

    for (size_t k = 0; k < static_cast<size_t>(total_recv); k++) {
        size_t dest = keep_indices.size() + k;
        const auto& p = recv_buf[k];
        new_ps.x[dest] = p.x;
        new_ps.y[dest] = p.y;
        new_ps.z[dest] = p.z;
        new_ps.vx[dest] = p.vx;
        new_ps.vy[dest] = p.vy;
        new_ps.vz[dest] = p.vz;
        new_ps.ax[dest] = p.ax;
        new_ps.ay[dest] = p.ay;
        new_ps.az[dest] = p.az;
        new_ps.m[dest] = p.m;
        new_ps.morton[dest] = p.morton;
    }

    ps = std::move(new_ps);
#else
    (void)ps;
    (void)global_bbox;
#endif
}

std::vector<RemoteMultipole> MpiContext::exchange_multipoles(const std::vector<RemoteMultipole>& local_nodes) const {
    std::vector<RemoteMultipole> multipoles;
#ifdef ASTRO_ENABLE_MPI
    if (!enabled || size <= 1) return local_nodes;

    int local_count = static_cast<int>(local_nodes.size());
    std::vector<int> recv_counts(size, 0);
    MPI_Allgather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> displs(size, 0);
    int total_nodes = recv_counts[0];
    for (int r = 1; r < size; r++) {
        displs[r] = displs[r - 1] + recv_counts[r - 1];
        total_nodes += recv_counts[r];
    }

    if (total_nodes == 0) return multipoles;

    multipoles.resize(total_nodes);

    std::vector<int> recv_bytes(size), displ_bytes(size);
    for (int r = 0; r < size; r++) {
        recv_bytes[r] = recv_counts[r] * static_cast<int>(sizeof(RemoteMultipole));
        displ_bytes[r] = displs[r] * static_cast<int>(sizeof(RemoteMultipole));
    }
    int my_send_bytes = local_count * static_cast<int>(sizeof(RemoteMultipole));

    MPI_Allgatherv(local_nodes.data(), my_send_bytes, MPI_BYTE,
                   multipoles.data(), recv_bytes.data(), displ_bytes.data(), MPI_BYTE,
                   MPI_COMM_WORLD);
#else
    (void)local_nodes;
#endif
    return multipoles;
}

std::vector<GlobalParticle> MpiContext::gather_all_particles(const ParticleSystem& ps, std::vector<int>& out_displs) const {
    std::vector<GlobalParticle> all_particles;
    out_displs.assign(size, 0);
#ifdef ASTRO_ENABLE_MPI
    if (!enabled || size <= 1) {
        all_particles.resize(ps.count);
        for (size_t i = 0; i < ps.count; i++) {
            all_particles[i] = {ps.x[i], ps.y[i], ps.z[i], ps.m[i]};
        }
        return all_particles;
    }

    int local_n = static_cast<int>(ps.count);
    std::vector<int> counts(size, 0);
    MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int total_n = counts[0];
    for (int r = 1; r < size; r++) {
        out_displs[r] = out_displs[r - 1] + counts[r - 1];
        total_n += counts[r];
    }

    std::vector<GlobalParticle> local_gp(ps.count);
    for (size_t i = 0; i < ps.count; i++) {
        local_gp[i] = {ps.x[i], ps.y[i], ps.z[i], ps.m[i]};
    }

    all_particles.resize(total_n);

    std::vector<int> recv_bytes(size), displ_bytes(size);
    for (int r = 0; r < size; r++) {
        recv_bytes[r] = counts[r] * static_cast<int>(sizeof(GlobalParticle));
        displ_bytes[r] = out_displs[r] * static_cast<int>(sizeof(GlobalParticle));
    }
    int send_bytes = local_n * static_cast<int>(sizeof(GlobalParticle));

    MPI_Allgatherv(local_gp.data(), send_bytes, MPI_BYTE,
                   all_particles.data(), recv_bytes.data(), displ_bytes.data(), MPI_BYTE,
                   MPI_COMM_WORLD);
#else
    all_particles.resize(ps.count);
    for (size_t i = 0; i < ps.count; i++) {
        all_particles[i] = {ps.x[i], ps.y[i], ps.z[i], ps.m[i]};
    }
    (void)ps;
    (void)out_displs;
#endif
    return all_particles;
}

} // namespace astro
