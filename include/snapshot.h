#ifndef SNAPSHOT_H
#define SNAPSHOT_H

#include <stdint.h>

#define SNAPSHOT_MAGIC 0x41535452 // "ASTR"

#define SNAPSHOT_FLAG_HAS_MASS 0x1

typedef struct {
    uint32_t magic;        // 0x41535452
    uint32_t step;         // Simulation step
    uint32_t n;            // Particle count
    float sim_time;        // Elapsed simulation time
    float dt;              // Time step dt
    uint32_t flags;        // Bit flags (e.g. SNAPSHOT_FLAG_HAS_MASS)
    uint8_t reserved[8];   // 32-byte total alignment padding
} SnapshotHeader;

// Write binary snapshot to disk: header + x[n] + y[n] + z[n] + vx[n] + vy[n] + vz[n] + optional m[n]
int snapshot_write(const char *filepath,
                   const float *x, const float *y, const float *z,
                   const float *vx, const float *vy, const float *vz,
                   const float *m,
                   int n, int step, float time, float dt);

// Read binary snapshot from disk into buffers
int snapshot_read(const char *filepath,
                  float *x, float *y, float *z,
                  float *vx, float *vy, float *vz,
                  float *m,
                  int max_n, SnapshotHeader *header);

#endif // SNAPSHOT_H
