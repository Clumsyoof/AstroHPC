#include "snapshot.h"
#include <stdio.h>
#include <string.h>

int snapshot_write(const char *filepath,
                   const float *x, const float *y, const float *z,
                   const float *vx, const float *vy, const float *vz,
                   const float *m,
                   int n, int step, float time, float dt) {
    if (!filepath || !x || !y || !z || n <= 0) return -1;

    FILE *f = fopen(filepath, "wb");
    if (!f) return -1;

    SnapshotHeader header;
    memset(&header, 0, sizeof(header));
    header.magic = SNAPSHOT_MAGIC;
    header.step = (uint32_t)step;
    header.n = (uint32_t)n;
    header.sim_time = time;
    header.dt = dt;
    header.flags = (m != NULL) ? SNAPSHOT_FLAG_HAS_MASS : 0;

    if (fwrite(&header, sizeof(SnapshotHeader), 1, f) != 1) {
        fclose(f);
        return -1;
    }

    size_t count = (size_t)n;
    if (fwrite(x, sizeof(float), count, f) != count ||
        fwrite(y, sizeof(float), count, f) != count ||
        fwrite(z, sizeof(float), count, f) != count) {
        fclose(f);
        return -1;
    }

    if (vx && vy && vz) {
        if (fwrite(vx, sizeof(float), count, f) != count ||
            fwrite(vy, sizeof(float), count, f) != count ||
            fwrite(vz, sizeof(float), count, f) != count) {
            fclose(f);
            return -1;
        }
    }

    if (m) {
        if (fwrite(m, sizeof(float), count, f) != count) {
            fclose(f);
            return -1;
        }
    }

    fclose(f);
    return 0;
}

int snapshot_read(const char *filepath,
                  float *x, float *y, float *z,
                  float *vx, float *vy, float *vz,
                  float *m,
                  int max_n, SnapshotHeader *out_header) {
    if (!filepath || !x || !y || !z || max_n <= 0) return -1;

    FILE *f = fopen(filepath, "rb");
    if (!f) return -1;

    SnapshotHeader header;
    if (fread(&header, sizeof(SnapshotHeader), 1, f) != 1) {
        fclose(f);
        return -1;
    }

    if (header.magic != SNAPSHOT_MAGIC) {
        fclose(f);
        return -1;
    }

    int to_read = (int)header.n;
    if (to_read > max_n) to_read = max_n;

    size_t count = (size_t)to_read;
    if (fread(x, sizeof(float), count, f) != count ||
        fread(y, sizeof(float), count, f) != count ||
        fread(z, sizeof(float), count, f) != count) {
        fclose(f);
        return -1;
    }

    // Attempt to read velocities if available in the file
    if (vx && vy && vz) {
        if (fread(vx, sizeof(float), count, f) != count ||
            fread(vy, sizeof(float), count, f) != count ||
            fread(vz, sizeof(float), count, f) != count) {
            // Zero out if not present in snapshot
            memset(vx, 0, count * sizeof(float));
            memset(vy, 0, count * sizeof(float));
            memset(vz, 0, count * sizeof(float));
        }
    }

    // Read masses if present in snapshot, or default to 1.0f
    if (m) {
        if ((header.flags & SNAPSHOT_FLAG_HAS_MASS) && fread(m, sizeof(float), count, f) == count) {
            // Successfully loaded individual masses
        } else {
            for (size_t i = 0; i < count; i++) {
                m[i] = 1.0f;
            }
        }
    }

    if (out_header) {
        *out_header = header;
    }

    fclose(f);
    return to_read;
}
