#ifndef ENGINE_H
#define ENGINE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n;
    int step;
    int active_nodes;
    float dt;
    float theta;
    float eps_sq;
    double step_time_ms;
    double fps;
    double kinetic_energy;
    double potential_energy;
    double total_energy;
} SimStats;

// Initialize simulation with galaxy disk preset
void sim_init_disk(int n, float radius, float central_mass, float disk_mass);

// Initialize simulation with 3-body preset
void sim_init_three_body(void);

// Advance simulation by 1 step
void sim_step(float dt, float theta, float eps_sq);

// Get current engine telemetry and statistics
SimStats sim_get_stats(void);

// Get body positions (projected into out_x and out_y arrays, up to max_count)
int sim_get_positions_2d(float *out_x, float *out_y, int max_count);

#ifdef __cplusplus
}
#endif

#endif // ENGINE_H
