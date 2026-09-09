#ifndef SNAPSHOT_HPP
#define SNAPSHOT_HPP

#include "particles.hpp"
#include <cstdio>
#include <string>

extern "C" {
#include "snapshot.h"
}

namespace astro {

inline int write_snapshot(const std::string& filepath, const ParticleSystem& ps,
                          int step, float time, float dt) {
    return snapshot_write(filepath.c_str(),
                          ps.x.data(), ps.y.data(), ps.z.data(),
                          ps.vx.data(), ps.vy.data(), ps.vz.data(),
                          ps.m.data(),
                          static_cast<int>(ps.count), step, time, dt);
}

inline bool read_snapshot(const std::string& filepath, ParticleSystem& ps,
                          SnapshotHeader* out_header = nullptr) {
    SnapshotHeader header;
    FILE* f = std::fopen(filepath.c_str(), "rb");
    if (!f) return false;
    if (std::fread(&header, sizeof(SnapshotHeader), 1, f) != 1) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);

    if (header.magic != SNAPSHOT_MAGIC) return false;

    ps.resize(header.n);
    int read_count = snapshot_read(filepath.c_str(),
                                   ps.x.data(), ps.y.data(), ps.z.data(),
                                   ps.vx.data(), ps.vy.data(), ps.vz.data(),
                                   ps.m.data(),
                                   static_cast<int>(header.n), &header);
    if (read_count < 0) return false;
    if (out_header) *out_header = header;
    return true;
}

} // namespace astro

#endif // SNAPSHOT_HPP
