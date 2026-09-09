#include <stdio.h>
#include <math.h>

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    #include <stdalign.h>
#elif defined(__GNUC__) || defined(__clang__)
    #define alignas(x) __attribute__((aligned(x)))
#else
    #define alignas(x)
#endif

#define MAX_BODIES 16384
#define G          1.0f
#define EPSILON_SQ 1e-2f
#define DT         0.01f

typedef struct {
    alignas(64) float x[MAX_BODIES];
    alignas(64) float y[MAX_BODIES];
    alignas(64) float z[MAX_BODIES];
    alignas(64) float vx[MAX_BODIES];
    alignas(64) float vy[MAX_BODIES];
    alignas(64) float vz[MAX_BODIES];
    alignas(64) float m[MAX_BODIES];
} Particles;

static Particles sys;

void step_simulation(
    float *restrict px,  float *restrict py,  float *restrict pz,
    float *restrict pvx, float *restrict pvy, float *restrict pvz,
    const float *restrict pm,
    const int n, const float dt
) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float ax = 0.0f;
        float ay = 0.0f;
        float az = 0.0f;

        const float xi = px[i];
        const float yi = py[i];
        const float zi = pz[i];

        // Inner loop: linear contiguous stream through memory
        #pragma GCC ivdep
        for (int j = 0; j < n; j++) {
            const float dx = px[j] - xi;
            const float dy = py[j] - yi;
            const float dz = pz[j] - zi;

            const float dist_sq = dx*dx + dy*dy + dz*dz + EPSILON_SQ;

            // Reciprocal square root optimization: 1 / (dist_sq)^(3/2)
            const float inv_dist = 1.0f / sqrtf(dist_sq);
            const float inv_cube = inv_dist * inv_dist * inv_dist;

            const float s = G * pm[j] * inv_cube;

            ax += s * dx;
            ay += s * dy;
            az += s * dz;
        }

        // Fused integrate step: update velocities and positions directly
        pvx[i] += ax * dt;
        pvy[i] += ay * dt;
        pvz[i] += az * dt;

        px[i]  += pvx[i] * dt;
        py[i]  += pvy[i] * dt;
        pz[i]  += pvz[i] * dt;
    }
}

int main(void) {
    const int N = 3;

    // Direct static initialization
    sys.x[0] = 0.0f;  sys.y[0] = 0.0f;  sys.z[0] = 0.0f;
    sys.vx[0] = 0.0f; sys.vy[0] = 0.0f; sys.vz[0] = 0.0f;
    sys.m[0] = 1000.0f;

    sys.x[1] = 5.0f;  sys.y[1] = 0.0f;  sys.z[1] = 0.0f;
    sys.vx[1] = 0.0f; sys.vy[1] = 14.0f; sys.vz[1] = 0.0f;
    sys.m[1] = 1.0f;

    sys.x[2] = 10.0f; sys.y[2] = 0.0f;  sys.z[2] = 0.0f;
    sys.vx[2] = 0.0f; sys.vy[2] = 10.0f; sys.vz[2] = 0.0f;
    sys.m[2] = 1.0f;

    for (int step = 0; step < 1000; step++) {
        step_simulation(
            sys.x, sys.y, sys.z,
            sys.vx, sys.vy, sys.vz,
            sys.m, N, DT
        );

        if (step<=1000) {
            printf("Step %3d | Body 1 Pos: (%6.2f, %6.2f) | Body 2 Pos: (%6.2f, %6.2f)\n",
                   step, sys.x[1], sys.y[1], sys.x[2], sys.y[2]);
        }
    }

    return 0;
}
