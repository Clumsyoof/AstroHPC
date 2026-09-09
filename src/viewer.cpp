#include <raylib.h>
#include <raymath.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <dirent.h>
#include <string>
#include <vector>
#include <algorithm>

#include "config.h"
#include "particles.hpp"
#include "backend/cpu_backend.hpp"
#include "snapshot.hpp"

#define MAX_SNAPSHOTS 2048

static astro::ParticleSystem g_sys;
static astro::CpuBackend g_backend;

// Snapshot playback state
struct SnapshotManager {
    char files[MAX_SNAPSHOTS][512];
    int count;
    int current_index;
    float playback_speed;
    float time_accumulator;
    bool is_playing;
    SnapshotHeader current_header;
};

static int compare_strings(const void *a, const void *b) {
    return strcmp((const char *)a, (const char *)b);
}

static int scan_snapshots(const char *dir_path, SnapshotManager *sm) {
    sm->count = 0;
    sm->current_index = 0;
    sm->playback_speed = 1.0f;
    sm->time_accumulator = 0.0f;
    sm->is_playing = true;

    DIR *d = opendir(dir_path);
    if (!d) return 0;

    struct dirent *dir;
    while ((dir = readdir(d)) != NULL && sm->count < MAX_SNAPSHOTS) {
        size_t len = strlen(dir->d_name);
        if (len > 4 && strcmp(dir->d_name + len - 4, ".bin") == 0) {
            snprintf(sm->files[sm->count], sizeof(sm->files[0]), "%s/%s", dir_path, dir->d_name);
            sm->count++;
        }
    }
    closedir(d);

    if (sm->count > 0) {
        qsort(sm->files, (size_t)sm->count, sizeof(sm->files[0]), compare_strings);
    }

    return sm->count;
}

static Color get_velocity_color(float speed, float max_speed) {
    float t = (max_speed > 1e-4f) ? (speed / max_speed) : 0.0f;
    if (t > 1.0f) t = 1.0f;

    if (t < 0.33f) {
        float f = t / 0.33f;
        return (Color){
            (unsigned char)(40 + f * 40),
            (unsigned char)(80 + f * 100),
            (unsigned char)(200 + f * 55),
            220
        };
    } else if (t < 0.66f) {
        float f = (t - 0.33f) / 0.33f;
        return (Color){
            (unsigned char)(80 + f * 150),
            (unsigned char)(180 + f * 60),
            (unsigned char)(255 - f * 100),
            230
        };
    } else {
        float f = (t - 0.66f) / 0.34f;
        return (Color){
            (unsigned char)(230 + f * 25),
            (unsigned char)(240 + f * 15),
            (unsigned char)(155 + f * 100),
            255
        };
    }
}

static void draw_octree_wires_recursive(const std::vector<astro::CpuOctNode>& nodes, int node_idx) {
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes.size())) return;
    const auto& node = nodes[node_idx];
    if (node.mass <= 0.0f) return;

    float size = node.half_size * 2.0f;
    Vector3 center = { node.cx, node.cy, node.cz };
    DrawCubeWires(center, size, size, size, (Color){ 60, 80, 120, 40 });

    for (int i = 0; i < 8; i++) {
        if (node.children[i] != -1) {
            draw_octree_wires_recursive(nodes, node.children[i]);
        }
    }
}

int main(int argc, char **argv) {
    const int screenWidth = 1280;
    const int screenHeight = 720;

    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
    InitWindow(screenWidth, screenHeight, "astrohpc - 3D N-Body Viewer");

    float cam_distance = 220.0f;
    float cam_yaw = 0.8f;
    float cam_pitch = 0.5f;
    Vector3 cam_target = { 0.0f, 0.0f, 0.0f };

    Camera3D camera{};
    camera.target = cam_target;
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    camera.position.x = cam_target.x + cam_distance * cosf(cam_pitch) * sinf(cam_yaw);
    camera.position.y = cam_target.y + cam_distance * sinf(cam_pitch);
    camera.position.z = cam_target.z + cam_distance * cosf(cam_pitch) * cosf(cam_yaw);

    SetTargetFPS(60);

    bool snapshot_mode = false;
    const char *csv_file = NULL;
    SnapshotManager sm{};

    int n_bodies = 4096;
    bool show_octree = false;
    bool live_paused = false;
    float live_speed = 1.0f;
    float sim_g = DEFAULT_G;
    int g_custom = 0;
    float sim_dt = DEFAULT_DT;
    int dt_custom = 0;
    float sim_eps_sq = DEFAULT_EPSILON_SQ;
    int eps_custom = 0;
    float sim_theta = DEFAULT_THETA;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-g") == 0 || strcmp(argv[i], "--g") == 0 || strcmp(argv[i], "--grav") == 0) && i + 1 < argc) {
            sim_g = (float)atof(argv[++i]);
            g_custom = 1;
        } else if (strcmp(argv[i], "--real") == 0 || strcmp(argv[i], "--irl") == 0) {
            sim_g = G_IRL_ASTRO;
            g_custom = 1;
        } else if ((strcmp(argv[i], "-dt") == 0 || strcmp(argv[i], "-d") == 0) && i + 1 < argc) {
            sim_dt = (float)atof(argv[++i]);
            dt_custom = 1;
        } else if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) {
            sim_eps_sq = (float)atof(argv[++i]);
            eps_custom = 1;
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            sim_theta = (float)atof(argv[++i]);
        } else if (argv[i][0] != '-') {
            size_t len = strlen(argv[i]);
            if (len > 4 && strcmp(argv[i] + len - 4, ".csv") == 0) {
                if (g_sys.load_csv(argv[i])) {
                    n_bodies = static_cast<int>(g_sys.count);
                    csv_file = argv[i];
                    printf("Loaded real astronomical dataset: %s (%d bodies)\n", argv[i], n_bodies);
                } else {
                    fprintf(stderr, "Failed to load CSV: %s\n", argv[i]);
                }
            } else if (scan_snapshots(argv[i], &sm) > 0) {
                snapshot_mode = true;
                astro::read_snapshot(sm.files[0], g_sys, &sm.current_header);
                n_bodies = static_cast<int>(g_sys.count);
            } else {
                if (g_sys.load_csv(argv[i])) {
                    n_bodies = static_cast<int>(g_sys.count);
                    csv_file = argv[i];
                    printf("Loaded real astronomical dataset: %s (%d bodies)\n", argv[i], n_bodies);
                } else {
                    printf("No snapshots or valid CSV found in %s, falling back to Live Mode.\n", argv[i]);
                }
            }
        }
    }

    if (csv_file) {
        if (!g_custom) sim_g = g_sys.default_g();
        if (!eps_custom) sim_eps_sq = g_sys.default_eps_sq();
        if (!dt_custom) sim_dt = g_sys.default_dt();
        if (g_sys.units == astro::DatasetUnits::SolarSystem) {
            cam_distance = 6.0f;
            cam_pitch = 0.55f;
            live_speed = 3.0f;
        } else {
            cam_distance = 45.0f;
        }
    }

    std::vector<std::vector<Vector3>> trails;
    if (g_sys.units == astro::DatasetUnits::SolarSystem) {
        trails.resize(g_sys.count);
    }

    if (!snapshot_mode) {
        if (!csv_file) {
            g_sys.init_disk(n_bodies, 80.0f, 800.0f, 150.0f);
        }
        g_backend.compute_forces(g_sys, sim_theta, sim_g, sim_eps_sq);
    }

    int track_target = 0; // Solar system mode: 0 = Sun, 1 = Mercury, etc. -1 = Free Pan

    while (!WindowShouldClose()) {
        // Camera presets and tracking for Solar System
        if (g_sys.units == astro::DatasetUnits::SolarSystem) {
            if (IsKeyPressed(KEY_ONE)) {
                // View Inner Planets (Sun, Mercury, Venus, Earth, Mars)
                track_target = 0;
                cam_distance = 3.5f;
                cam_pitch = 0.55f;
            }
            if (IsKeyPressed(KEY_TWO)) {
                // View Full Solar System (out to Pluto)
                track_target = 0;
                cam_distance = 55.0f;
                cam_pitch = 0.75f;
            }
            if (IsKeyPressed(KEY_TAB)) {
                // Cycle tracking through planets
                track_target = (track_target + 1) % static_cast<int>(g_sys.count);
                if (track_target != 0 && cam_distance > 5.0f) {
                    cam_distance = 1.8f;
                }
            }
            if (IsKeyPressed(KEY_C)) {
                // Re-center on Sun
                track_target = 0;
            }
        }

        // Left Click: Orbit
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            Vector2 delta = GetMouseDelta();
            cam_yaw -= delta.x * 0.005f;
            cam_pitch += delta.y * 0.005f;
            if (cam_pitch > 1.55f) cam_pitch = 1.55f;
            if (cam_pitch < -1.55f) cam_pitch = -1.55f;
        }

        // Right/Middle Click: Pan
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT) || IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
            track_target = -1; // disable tracking on manual pan
            Vector2 delta = GetMouseDelta();
            float sin_yaw = sinf(cam_yaw);
            float cos_yaw = cosf(cam_yaw);
            Vector3 right = { cos_yaw, 0.0f, -sin_yaw };
            float pan_speed = cam_distance * 0.0015f;
            cam_target.x -= right.x * delta.x * pan_speed;
            cam_target.y += delta.y * pan_speed;
            cam_target.z -= right.z * delta.x * pan_speed;
        }

        // Mouse Wheel Zoom
        float min_cam_dist = (g_sys.units == astro::DatasetUnits::SolarSystem) ? 0.05f : 5.0f;
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            cam_distance -= wheel * (cam_distance * 0.08f);
            if (cam_distance < min_cam_dist) cam_distance = min_cam_dist;
            if (cam_distance > 5000.0f) cam_distance = 5000.0f;
        }

        if (track_target >= 0 && track_target < static_cast<int>(g_sys.count)) {
            cam_target = (Vector3){ g_sys.x[track_target], g_sys.y[track_target], g_sys.z[track_target] };
            if (track_target == 4 && g_sys.count > 4) { // Moon tracking offset
                float mdx = g_sys.x[4] - g_sys.x[3];
                float mdy = g_sys.y[4] - g_sys.y[3];
                float mdz = g_sys.z[4] - g_sys.z[3];
                float md = sqrtf(mdx*mdx + mdy*mdy + mdz*mdz);
                if (md > 1e-5f) {
                    cam_target.x = g_sys.x[3] + (mdx / md) * 0.045f;
                    cam_target.y = g_sys.y[3] + (mdy / md) * 0.045f;
                    cam_target.z = g_sys.z[3] + (mdz / md) * 0.045f;
                }
            }
        }

        camera.target = cam_target;
        camera.position.x = cam_target.x + cam_distance * cosf(cam_pitch) * sinf(cam_yaw);
        camera.position.y = cam_target.y + cam_distance * sinf(cam_pitch);
        camera.position.z = cam_target.z + cam_distance * cosf(cam_pitch) * cosf(cam_yaw);

        // Input controls
        if (IsKeyPressed(KEY_SPACE)) {
            if (snapshot_mode) sm.is_playing = !sm.is_playing;
            else live_paused = !live_paused;
        }

        if (IsKeyPressed(KEY_R)) {
            for (auto& tr : trails) tr.clear();
            if (snapshot_mode) {
                sm.current_index = 0;
                astro::read_snapshot(sm.files[0], g_sys, &sm.current_header);
                n_bodies = static_cast<int>(g_sys.count);
            } else {
                if (csv_file) {
                    g_sys.load_csv(csv_file);
                    n_bodies = static_cast<int>(g_sys.count);
                } else {
                    g_sys.init_disk(n_bodies, 80.0f, 800.0f, 150.0f);
                }
                g_backend.compute_forces(g_sys, sim_theta, sim_g, sim_eps_sq);
            }
        }

        if (IsKeyPressed(KEY_O)) {
            show_octree = !show_octree;
        }

        if (snapshot_mode) {
            if (IsKeyPressed(KEY_RIGHT) && sm.current_index < sm.count - 1) {
                sm.current_index++;
                astro::read_snapshot(sm.files[sm.current_index], g_sys, &sm.current_header);
                n_bodies = static_cast<int>(g_sys.count);
            }
            if (IsKeyPressed(KEY_LEFT) && sm.current_index > 0) {
                sm.current_index--;
                astro::read_snapshot(sm.files[sm.current_index], g_sys, &sm.current_header);
                n_bodies = static_cast<int>(g_sys.count);
            }
            if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_EQUAL)) {
                sm.playback_speed *= 1.5f;
                if (sm.playback_speed > 8.0f) sm.playback_speed = 8.0f;
            }
            if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_MINUS)) {
                sm.playback_speed /= 1.5f;
                if (sm.playback_speed < 0.25f) sm.playback_speed = 0.25f;
            }

            if (sm.is_playing && sm.count > 1) {
                sm.time_accumulator += GetFrameTime() * sm.playback_speed;
                float interval = 0.05f;
                if (sm.time_accumulator >= interval) {
                    sm.time_accumulator = 0.0f;
                    sm.current_index = (sm.current_index + 1) % sm.count;
                    astro::read_snapshot(sm.files[sm.current_index], g_sys, &sm.current_header);
                    n_bodies = static_cast<int>(g_sys.count);
                }
            }
        } else {
            if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_EQUAL)) {
                live_speed *= 1.5f;
                if (live_speed > 8.0f) live_speed = 8.0f;
            }
            if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_MINUS)) {
                live_speed /= 1.5f;
                if (live_speed < 0.25f) live_speed = 0.25f;
            }

            if (!live_paused) {
                int substeps = (g_sys.units == astro::DatasetUnits::SolarSystem)
                               ? static_cast<int>(live_speed * 4.0f)
                               : ((live_speed >= 2.0f) ? static_cast<int>(live_speed) : 1);
                if (substeps < 1) substeps = 1;
                float dt_step = sim_dt * (live_speed / static_cast<float>(substeps));
                for (int s = 0; s < substeps; s++) {
                    g_backend.compute_forces(g_sys, sim_theta, sim_g, sim_eps_sq);
                    g_sys.integrate_symplectic(dt_step);
                }
                if (g_sys.units == astro::DatasetUnits::SolarSystem) {
                    if (trails.size() != g_sys.count) trails.resize(g_sys.count);
                    for (size_t i = 1; i < g_sys.count; i++) {
                        Vector3 pt = { g_sys.x[i], g_sys.y[i], g_sys.z[i] };
                        if (i == 4 && g_sys.count > 4) { // Moon visual offset from Earth
                            float mdx = g_sys.x[4] - g_sys.x[3];
                            float mdy = g_sys.y[4] - g_sys.y[3];
                            float mdz = g_sys.z[4] - g_sys.z[3];
                            float md = sqrtf(mdx * mdx + mdy * mdy + mdz * mdz);
                            if (md > 1e-5f) {
                                pt.x = g_sys.x[3] + (mdx / md) * 0.045f;
                                pt.y = g_sys.y[3] + (mdy / md) * 0.045f;
                                pt.z = g_sys.z[3] + (mdz / md) * 0.045f;
                            }
                        }
                        trails[i].push_back(pt);
                        if (trails[i].size() > 400) {
                            trails[i].erase(trails[i].begin());
                        }
                    }
                }
            } else {
                if (IsKeyPressed(KEY_RIGHT)) {
                    g_backend.compute_forces(g_sys, sim_theta, sim_g, sim_eps_sq);
                    g_sys.integrate_symplectic(sim_dt);
                }
                if (IsKeyPressed(KEY_LEFT)) {
                    g_sys.integrate_symplectic_reverse_pos(sim_dt);
                    g_backend.compute_forces(g_sys, sim_theta, sim_g, sim_eps_sq);
                    g_sys.integrate_symplectic_reverse_vel(sim_dt);
                }
            }
        }

        BeginDrawing();
        ClearBackground((Color){ 10, 10, 16, 255 });

        BeginMode3D(camera);

        static const std::vector<Color> solar_colors = {
            (Color){ 255, 230, 70, 255 },  // 0: Sun
            (Color){ 190, 190, 195, 255 },  // 1: Mercury
            (Color){ 235, 215, 170, 255 },  // 2: Venus
            (Color){ 90, 160, 245, 255 },   // 3: Earth
            (Color){ 180, 180, 190, 255 },  // 4: Moon
            (Color){ 235, 95, 65, 255 },    // 5: Mars
            (Color){ 225, 175, 120, 255 },  // 6: Jupiter
            (Color){ 235, 215, 150, 255 },  // 7: Saturn
            (Color){ 140, 220, 235, 255 },  // 8: Uranus
            (Color){ 70, 110, 245, 255 },   // 9: Neptune
            (Color){ 175, 155, 145, 255 },  // 10: Pluto
        };

        // Realistic proportions in AU: Mercury perihelion is ~0.307 AU, so Sun at 0.080 AU leaves vast empty space
        static const std::vector<float> solar_radii = {
            0.080f,  // 0: Sun
            0.016f,  // 1: Mercury (distance 0.387 AU)
            0.024f,  // 2: Venus (distance 0.723 AU)
            0.026f,  // 3: Earth (distance 1.000 AU)
            0.010f,  // 4: Moon (orbiting Earth)
            0.020f,  // 5: Mars (distance 1.524 AU)
            0.065f,  // 6: Jupiter (distance 5.204 AU)
            0.055f,  // 7: Saturn (distance 9.582 AU)
            0.040f,  // 8: Uranus (distance 19.20 AU)
            0.040f,  // 9: Neptune (distance 30.05 AU)
            0.015f   // 10: Pluto (distance 39.48 AU)
        };

        if (g_sys.units == astro::DatasetUnits::SolarSystem) {
            // Draw Keplerian orbit trails
            for (size_t i = 1; i < trails.size(); i++) {
                const auto& tr = trails[i];
                if (tr.size() < 2) continue;
                Color col = (i < solar_colors.size()) ? solar_colors[i] : SKYBLUE;
                for (size_t p = 1; p < tr.size(); p++) {
                    float alpha = (float)p / (float)tr.size();
                    DrawLine3D(tr[p - 1], tr[p], ColorAlpha(col, alpha * 0.55f));
                }
            }

            // Draw solar system bodies
            for (size_t i = 0; i < g_sys.count; i++) {
                Vector3 pos = { g_sys.x[i], g_sys.y[i], g_sys.z[i] };
                if (i == 4 && g_sys.count > 4) { // Moon visual offset
                    float mdx = g_sys.x[4] - g_sys.x[3];
                    float mdy = g_sys.y[4] - g_sys.y[3];
                    float mdz = g_sys.z[4] - g_sys.z[3];
                    float md = sqrtf(mdx * mdx + mdy * mdy + mdz * mdz);
                    if (md > 1e-5f) {
                        pos.x = g_sys.x[3] + (mdx / md) * 0.045f;
                        pos.y = g_sys.y[3] + (mdy / md) * 0.045f;
                        pos.z = g_sys.z[3] + (mdz / md) * 0.045f;
                    }
                }
                Color c = (i < solar_colors.size()) ? solar_colors[i] : WHITE;
                float r = (i < solar_radii.size()) ? solar_radii[i] : 0.02f;

                if (i == 0) {
                    // Sun: glowing star, cleanly bounded within 0.096 AU
                    DrawSphere(pos, r, c);
                    DrawSphereWires(pos, r * 1.20f, 10, 10, (Color){ 255, 180, 30, 140 });
                } else {
                    DrawSphere(pos, r, c);
                    if (i == 7) {
                        // Saturn ring
                        DrawCircle3D(pos, r * 2.1f, (Vector3){ 0, 1, 0 }, 90.0f, ColorAlpha(c, 0.5f));
                        DrawCircle3D(pos, r * 2.5f, (Vector3){ 0, 1, 0 }, 90.0f, ColorAlpha(c, 0.35f));
                    }
                }
            }
        } else {
            float max_speed = 5.0f;
            for (size_t i = 0; i < g_sys.count; i += 16) {
                float s = sqrtf(g_sys.vx[i]*g_sys.vx[i] + g_sys.vy[i]*g_sys.vy[i] + g_sys.vz[i]*g_sys.vz[i]);
                if (s > max_speed) max_speed = s;
            }

            for (size_t i = 0; i < g_sys.count; i++) {
                Vector3 pos = { g_sys.x[i], g_sys.y[i], g_sys.z[i] };
                if (i == 0 && g_sys.m[0] > 50.0f) {
                    DrawSphere(pos, 2.0f, (Color){ 255, 240, 160, 255 });
                    DrawSphereWires(pos, 2.2f, 8, 8, (Color){ 255, 200, 50, 180 });
                } else {
                    float spd = sqrtf(g_sys.vx[i]*g_sys.vx[i] + g_sys.vy[i]*g_sys.vy[i] + g_sys.vz[i]*g_sys.vz[i]);
                    Color c = get_velocity_color(spd, max_speed);
                    float m = g_sys.m[i];
                    float sz = (m > 0.5f) ? (0.5f + cbrtf(m) * 0.25f) : 0.45f;
                    if (sz > 2.5f) sz = 2.5f;
                    DrawCube(pos, sz, sz, sz, c);
                }
            }
        }

        if (!snapshot_mode && show_octree) {
            draw_octree_wires_recursive(g_backend.get_nodes(), 0);
        }

        EndMode3D();

        // 2D overlays: planetary markers and labels
        if (g_sys.units == astro::DatasetUnits::SolarSystem) {
            for (size_t i = 0; i < g_sys.count; i++) {
                Vector3 pos = { g_sys.x[i], g_sys.y[i], g_sys.z[i] };
                if (i == 4 && g_sys.count > 4) { // Moon visual offset
                    float mdx = g_sys.x[4] - g_sys.x[3];
                    float mdy = g_sys.y[4] - g_sys.y[3];
                    float mdz = g_sys.z[4] - g_sys.z[3];
                    float md = sqrtf(mdx * mdx + mdy * mdy + mdz * mdz);
                    if (md > 1e-5f) {
                        pos.x = g_sys.x[3] + (mdx / md) * 0.045f;
                        pos.y = g_sys.y[3] + (mdy / md) * 0.045f;
                        pos.z = g_sys.z[3] + (mdz / md) * 0.045f;
                    }
                }
                Vector2 scr = GetWorldToScreen(pos, camera);
                if (scr.x > 0 && scr.x < screenWidth && scr.y > 0 && scr.y < screenHeight) {
                    Color c = (i < solar_colors.size()) ? solar_colors[i] : WHITE;
                    float r = (i < solar_radii.size()) ? solar_radii[i] : 0.02f;

                    // Approximate screen-space radius of the 3D sphere
                    float dx = camera.position.x - pos.x;
                    float dy = camera.position.y - pos.y;
                    float dz = camera.position.z - pos.z;
                    float dist_to_cam = sqrtf(dx * dx + dy * dy + dz * dz);
                    float proj_radius = (dist_to_cam > 0.01f)
                        ? (r / dist_to_cam) * (screenHeight / (2.0f * tanf(camera.fovy * DEG2RAD * 0.5f)))
                        : 5.0f;

                    // If sphere is tiny on screen (< 3.5px), draw a crisp 2D point/halo so it never vanishes
                    if (proj_radius < 3.5f) {
                        DrawCircleV(scr, (i == 0) ? 4.5f : 2.5f, c);
                        if (i == 0) DrawCircleV(scr, 7.5f, ColorAlpha(c, 0.35f));
                    }

                    const char* name = (i < g_sys.names.size() && !g_sys.names[i].empty())
                                       ? g_sys.names[i].c_str() : "";
                    DrawText(name, (int)scr.x + 8, (int)scr.y - 6, 12, (Color){ 210, 225, 255, 210 });
                }
            }
        }

        int hud_h = (g_sys.units == astro::DatasetUnits::SolarSystem) ? 190 : ((csv_file != NULL) ? 175 : 155);
        DrawRectangle(15, 15, 320, hud_h, (Color){ 20, 20, 30, 210 });
        DrawRectangleLines(15, 15, 320, hud_h, (Color){ 60, 70, 90, 255 });

        DrawText("astrohpc", 30, 25, 20, (Color){ 140, 160, 255, 255 });

        if (snapshot_mode) {
            DrawText(TextFormat("Mode: REPLAY [%d / %d]", sm.current_index + 1, sm.count), 30, 52, 14, RAYWHITE);
            DrawText(TextFormat("State: %s (%.2fx speed)", sm.is_playing ? "PLAYING" : "PAUSED", sm.playback_speed), 30, 70, 14,
                     sm.is_playing ? GREEN : YELLOW);
            DrawText(TextFormat("Step: %u | Sim Time: %.2f", sm.current_header.step, sm.current_header.sim_time), 30, 88, 14, LIGHTGRAY);
            DrawText(TextFormat("Bodies: %d | FPS: %d", n_bodies, GetFPS()), 30, 106, 14, LIGHTGRAY);

            int barW = screenWidth - 60;
            int barH = 8;
            int barX = 30;
            int barY = screenHeight - 30;
            DrawRectangle(barX, barY, barW, barH, (Color){ 40, 40, 50, 200 });
            float progress = (sm.count > 1) ? ((float)sm.current_index / (float)(sm.count - 1)) : 1.0f;
            DrawRectangle(barX, barY, (int)(barW * progress), barH, (Color){ 100, 140, 255, 255 });
            DrawText("[Space] Pause | [Left/Right] Step | [Up/Down] Speed | [R] Rewind", 30, screenHeight - 50, 12, GRAY);
        } else {
            DrawText("Mode: LIVE SIMULATION", 30, 52, 14, RAYWHITE);
            DrawText(TextFormat("State: %s (%.2fx speed)", live_paused ? "PAUSED" : "RUNNING", live_speed), 30, 70, 14,
                     live_paused ? YELLOW : GREEN);
            DrawText(TextFormat("Bodies: %d | Nodes: %d | FPS: %d", n_bodies, g_backend.get_node_count(), GetFPS()), 30, 88, 14, LIGHTGRAY);
            DrawText(TextFormat("G: %.4f (%s)", sim_g,
                     (g_sys.units == astro::DatasetUnits::SolarSystem) ? "Solar AU/yr" :
                     ((fabsf(sim_g - G_IRL_ASTRO) < 1e-6f) ? "Galactic pc/Myr" : "custom")),
                     30, 106, 14,
                     (g_sys.units == astro::DatasetUnits::SolarSystem || fabsf(sim_g - G_IRL_ASTRO) < 1e-6f)
                     ? (Color){100, 220, 120, 255} : LIGHTGRAY);
            if (csv_file) {
                DrawText(TextFormat("Data: %s", GetFileName(csv_file)), 30, 124, 14, LIGHTGRAY);
                if (g_sys.units == astro::DatasetUnits::SolarSystem) {
                    const char* target_str = (track_target >= 0 && track_target < (int)g_sys.names.size() && !g_sys.names[track_target].empty())
                                             ? g_sys.names[track_target].c_str() : ((track_target == -1) ? "Free Pan" : "Sun");
                    DrawText(TextFormat("Target: %s (TAB to cycle)", target_str), 30, 142, 14, (Color){130, 210, 255, 255});
                    DrawText(TextFormat("Octree (O): %s", show_octree ? "ON" : "OFF"), 30, 160, 14, LIGHTGRAY);
                } else {
                    DrawText(TextFormat("Octree (O): %s", show_octree ? "ON" : "OFF"), 30, 142, 14, LIGHTGRAY);
                }
            } else {
                DrawText(TextFormat("Octree (O): %s", show_octree ? "ON" : "OFF"), 30, 124, 14, LIGHTGRAY);
            }
            if (g_sys.units == astro::DatasetUnits::SolarSystem) {
                DrawText("[1] Inner System | [2] Full System | [TAB] Track Body | [C] Center Sun", 30, screenHeight - 68, 12, (Color){150, 200, 255, 220});
            }
            DrawText("[Up/Down] Speed | [Space] Pause | [Left/Right] Step -/+ | [O] Octree", 30, screenHeight - 50, 12, GRAY);
            DrawText("[Left Click Drag] Orbit | [Right Click Drag] Pan | [Wheel] Zoom", 30, screenHeight - 32, 12, GRAY);
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
