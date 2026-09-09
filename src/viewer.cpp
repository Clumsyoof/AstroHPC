#include <raylib.h>
#include <raymath.h>
#include <rlgl.h>

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

static Color get_celestial_color(const std::string& name, size_t index, astro::DatasetUnits units, float speed, float max_speed) {
    if (!name.empty()) {
        std::string lower = name;
        for (char& c : lower) c = (char)tolower((unsigned char)c);

        if (lower.find("sun") != std::string::npos)     return (Color){ 255, 230, 70, 255 };
        if (lower.find("mercury") != std::string::npos) return (Color){ 190, 190, 195, 255 };
        if (lower.find("venus") != std::string::npos)   return (Color){ 235, 215, 170, 255 };
        if (lower.find("earth") != std::string::npos)   return (Color){ 90, 160, 245, 255 };
        if (lower.find("moon") != std::string::npos)    return (Color){ 195, 195, 205, 255 };
        if (lower.find("mars") != std::string::npos)    return (Color){ 235, 95, 65, 255 };
        if (lower.find("jupiter") != std::string::npos) return (Color){ 225, 175, 120, 255 };
        if (lower.find("saturn") != std::string::npos)  return (Color){ 235, 215, 150, 255 };
        if (lower.find("uranus") != std::string::npos)  return (Color){ 140, 220, 235, 255 };
        if (lower.find("neptune") != std::string::npos) return (Color){ 70, 110, 245, 255 };
        if (lower.find("pluto") != std::string::npos)   return (Color){ 175, 155, 145, 255 };
    }

    if (units == astro::DatasetUnits::SolarSystem) {
        static const Color default_solar_colors[] = {
            (Color){ 255, 230, 70, 255 },   // 0: Sun
            (Color){ 190, 190, 195, 255 },  // 1: Mercury
            (Color){ 235, 215, 170, 255 },  // 2: Venus
            (Color){ 90, 160, 245, 255 },   // 3: Earth
            (Color){ 195, 195, 205, 255 },  // 4: Moon
            (Color){ 235, 95, 65, 255 },    // 5: Mars
            (Color){ 225, 175, 120, 255 },  // 6: Jupiter
            (Color){ 235, 215, 150, 255 },  // 7: Saturn
            (Color){ 140, 220, 235, 255 },  // 8: Uranus
            (Color){ 70, 110, 245, 255 },   // 9: Neptune
            (Color){ 175, 155, 145, 255 },  // 10: Pluto
        };
        if (index < sizeof(default_solar_colors) / sizeof(default_solar_colors[0])) {
            return default_solar_colors[index];
        }
    }

    return get_velocity_color(speed, max_speed);
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
        } else if (strcmp(argv[i], "-dt") == 0 && i + 1 < argc) {
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
    }

    if (!snapshot_mode) {
        if (!csv_file) {
            g_sys.init_disk(n_bodies, 80.0f, 800.0f, 150.0f);
        }
        g_backend.compute_forces(g_sys, sim_theta, sim_g, sim_eps_sq);
    }

    // Compute characteristic spatial scale of the dataset
    float max_r = 1.0f;
    for (size_t i = 0; i < g_sys.count; i++) {
        float r = sqrtf(g_sys.x[i] * g_sys.x[i] + g_sys.y[i] * g_sys.y[i] + g_sys.z[i] * g_sys.z[i]);
        if (r > max_r) max_r = r;
    }
    cam_distance = max_r * 1.5f;
    float min_cam_dist = fmaxf(max_r * 0.00002f, 0.0001f);
    float max_cam_dist = max_r * 4.0f;
    int tracked_idx = -1; // -1 = free camera target, >= 0 = locked to a particle
    static Vector2 mouse_click_pos = { 0, 0 };
    static bool mouse_was_pressed = false;

    std::vector<std::vector<Vector3>> trails;
    if (g_sys.units == astro::DatasetUnits::SolarSystem) {
        trails.resize(g_sys.count);
    }

    while (!WindowShouldClose()) {
        int screenWidth = GetScreenWidth();
        int screenHeight = GetScreenHeight();

        // 1. Orbit (Left Click Drag)
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            Vector2 delta = GetMouseDelta();
            cam_yaw -= delta.x * 0.005f;
            cam_pitch += delta.y * 0.005f;
            // Clamp pitch to avoid gimbal lock flip at poles
            if (cam_pitch > 1.45f) cam_pitch = 1.45f;
            if (cam_pitch < -1.45f) cam_pitch = -1.45f;
        }

        // 2. Click to Focus on Any Particle (Left Click Release without Drag)
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            mouse_click_pos = GetMousePosition();
            mouse_was_pressed = true;
        }
        if (mouse_was_pressed && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
            mouse_was_pressed = false;
            Vector2 release_pos = GetMousePosition();
            float drag_len = sqrtf((release_pos.x - mouse_click_pos.x) * (release_pos.x - mouse_click_pos.x) +
                                   (release_pos.y - mouse_click_pos.y) * (release_pos.y - mouse_click_pos.y));
            if (drag_len < 5.0f) {
                // Find closest particle in front of camera
                Vector3 cam_fwd = { camera.target.x - camera.position.x,
                                    camera.target.y - camera.position.y,
                                    camera.target.z - camera.position.z };
                int best_i = -1;
                float best_d = 24.0f;
                for (size_t i = 0; i < g_sys.count; i++) {
                    Vector3 p = { g_sys.x[i], g_sys.y[i], g_sys.z[i] };
                    Vector3 to_p = { p.x - camera.position.x, p.y - camera.position.y, p.z - camera.position.z };
                    if (to_p.x * cam_fwd.x + to_p.y * cam_fwd.y + to_p.z * cam_fwd.z <= 0.0f) continue;

                    Vector2 s = GetWorldToScreen(p, camera);
                    if (s.x > 0 && s.x < screenWidth && s.y > 0 && s.y < screenHeight) {
                        float d = sqrtf((release_pos.x - s.x) * (release_pos.x - s.x) + (release_pos.y - s.y) * (release_pos.y - s.y));
                        if (d < best_d) {
                            best_d = d;
                            best_i = static_cast<int>(i);
                        }
                    }
                }
                if (best_i >= 0) {
                    tracked_idx = best_i;
                    if (cam_distance > max_r * 0.15f) {
                        cam_distance = max_r * 0.05f;
                    }
                }
            }
        }

        // 3. Pan (Right Click / Middle Click Drag)
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT) || IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
            tracked_idx = -1; // Detach tracking on manual pan
            Vector2 delta = GetMouseDelta();
            float sin_yaw = sinf(cam_yaw);
            float cos_yaw = cosf(cam_yaw);
            Vector3 right = { cos_yaw, 0.0f, -sin_yaw };
            float pan_speed = fmaxf(cam_distance * 0.0015f, min_cam_dist * 0.5f);
            if (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT)) pan_speed *= 4.0f;
            cam_target.x -= right.x * delta.x * pan_speed;
            cam_target.y += delta.y * pan_speed;
            cam_target.z -= right.z * delta.x * pan_speed;
        }

        // 4. WASD Keyboard Fly/Pan Controls
        float key_speed = fmaxf(cam_distance * 0.015f, min_cam_dist);
        if (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT)) key_speed *= 3.0f;
        Vector3 cam_fwd_dir = { sinf(cam_yaw) * cosf(cam_pitch), -sinf(cam_pitch), cosf(cam_yaw) * cosf(cam_pitch) };
        Vector3 cam_rgt_dir = { cosf(cam_yaw), 0.0f, -sinf(cam_yaw) };
        bool key_moved = false;
        if (IsKeyDown(KEY_W)) { cam_target.x += cam_fwd_dir.x * key_speed; cam_target.y += cam_fwd_dir.y * key_speed; cam_target.z += cam_fwd_dir.z * key_speed; key_moved = true; }
        if (IsKeyDown(KEY_S)) { cam_target.x -= cam_fwd_dir.x * key_speed; cam_target.y -= cam_fwd_dir.y * key_speed; cam_target.z -= cam_fwd_dir.z * key_speed; key_moved = true; }
        if (IsKeyDown(KEY_A)) { cam_target.x -= cam_rgt_dir.x * key_speed; cam_target.y -= cam_rgt_dir.y * key_speed; cam_target.z -= cam_rgt_dir.z * key_speed; key_moved = true; }
        if (IsKeyDown(KEY_D)) { cam_target.x += cam_rgt_dir.x * key_speed; cam_target.y += cam_rgt_dir.y * key_speed; cam_target.z += cam_rgt_dir.z * key_speed; key_moved = true; }
        if (key_moved) tracked_idx = -1;

        // 5. Stable Exponential Zoom (Mouse Wheel)
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            cam_distance *= powf(0.88f, wheel);
            if (cam_distance < min_cam_dist) cam_distance = min_cam_dist;
            if (cam_distance > max_cam_dist) cam_distance = max_cam_dist;
        }

        // 6. Focus Key (F) on hovered particle, Center Key (C), and Cycle Key (TAB)
        if (IsKeyPressed(KEY_F)) {
            Vector2 mpos = GetMousePosition();
            Vector3 cam_fwd = { camera.target.x - camera.position.x,
                                camera.target.y - camera.position.y,
                                camera.target.z - camera.position.z };
            int best_i = -1;
            float best_d = 40.0f;
            for (size_t i = 0; i < g_sys.count; i++) {
                Vector3 p = { g_sys.x[i], g_sys.y[i], g_sys.z[i] };
                Vector3 to_p = { p.x - camera.position.x, p.y - camera.position.y, p.z - camera.position.z };
                if (to_p.x * cam_fwd.x + to_p.y * cam_fwd.y + to_p.z * cam_fwd.z <= 0.0f) continue;

                Vector2 s = GetWorldToScreen(p, camera);
                if (s.x > 0 && s.x < screenWidth && s.y > 0 && s.y < screenHeight) {
                    float d = sqrtf((mpos.x - s.x) * (mpos.x - s.x) + (mpos.y - s.y) * (mpos.y - s.y));
                    if (d < best_d) {
                        best_d = d;
                        best_i = static_cast<int>(i);
                    }
                }
            }
            if (best_i >= 0) {
                tracked_idx = best_i;
                if (cam_distance > max_r * 0.15f) {
                    cam_distance = max_r * 0.05f;
                }
            }
        }

        if (IsKeyPressed(KEY_C)) {
            tracked_idx = -1;
            cam_target = (Vector3){ 0.0f, 0.0f, 0.0f };
            cam_distance = max_r * 1.5f;
        }

        if (IsKeyPressed(KEY_TAB)) {
            if (g_sys.count > 0) {
                if (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT)) {
                    tracked_idx = (tracked_idx - 1 + static_cast<int>(g_sys.count)) % static_cast<int>(g_sys.count);
                } else {
                    tracked_idx = (tracked_idx + 1) % static_cast<int>(g_sys.count);
                }
                if (cam_distance > max_r * 0.15f) {
                    cam_distance = max_r * 0.05f;
                }
            }
        }

        // Input controls
        if (IsKeyPressed(KEY_SPACE)) {
            if (snapshot_mode) sm.is_playing = !sm.is_playing;
            else live_paused = !live_paused;
        }

        if (IsKeyPressed(KEY_R)) {
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
            for (auto& t : trails) t.clear();
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
                int substeps = (live_speed >= 2.0f) ? (int)live_speed : 1;
                float dt_step = sim_dt * (live_speed / (float)substeps);
                for (int s = 0; s < substeps; s++) {
                    g_backend.compute_forces(g_sys, sim_theta, sim_g, sim_eps_sq);
                    g_sys.integrate_symplectic(dt_step);
                }
                if (g_sys.units == astro::DatasetUnits::SolarSystem) {
                    if (trails.size() != g_sys.count) trails.resize(g_sys.count);
                    for (size_t i = 1; i < g_sys.count; i++) {
                        trails[i].push_back((Vector3){ g_sys.x[i], g_sys.y[i], g_sys.z[i] });
                        if (trails[i].size() > 500) trails[i].erase(trails[i].begin());
                    }
                }
            } else {
                if (IsKeyPressed(KEY_RIGHT)) {
                    g_backend.compute_forces(g_sys, sim_theta, sim_g, sim_eps_sq);
                    g_sys.integrate_symplectic(sim_dt);
                    if (g_sys.units == astro::DatasetUnits::SolarSystem) {
                        if (trails.size() != g_sys.count) trails.resize(g_sys.count);
                        for (size_t i = 1; i < g_sys.count; i++) {
                            trails[i].push_back((Vector3){ g_sys.x[i], g_sys.y[i], g_sys.z[i] });
                            if (trails[i].size() > 500) trails[i].erase(trails[i].begin());
                        }
                    }
                }
                if (IsKeyPressed(KEY_LEFT)) {
                    g_sys.integrate_symplectic_reverse_pos(sim_dt);
                    g_backend.compute_forces(g_sys, sim_theta, sim_g, sim_eps_sq);
                    g_sys.integrate_symplectic_reverse_vel(sim_dt);
                    if (g_sys.units == astro::DatasetUnits::SolarSystem) {
                        for (size_t i = 1; i < trails.size(); i++) {
                            if (!trails[i].empty()) trails[i].pop_back();
                        }
                    }
                }
            }
        }

        // Update tracking target and camera right before rendering so physics is in sync
        if (tracked_idx >= 0 && tracked_idx < static_cast<int>(g_sys.count)) {
            cam_target = (Vector3){ g_sys.x[tracked_idx], g_sys.y[tracked_idx], g_sys.z[tracked_idx] };
        }

        camera.target = cam_target;
        camera.position.x = cam_target.x + cam_distance * cosf(cam_pitch) * sinf(cam_yaw);
        camera.position.y = cam_target.y + cam_distance * sinf(cam_pitch);
        camera.position.z = cam_target.z + cam_distance * cosf(cam_pitch) * cosf(cam_yaw);

        // Dynamically adjust clipping planes to avoid clipping at near and far scales
        rlSetClipPlanes(fmaxf(cam_distance * 0.001f, 0.0001f), fmaxf(max_cam_dist * 3.0f, 5000.0f));

        BeginDrawing();
        ClearBackground((Color){ 10, 10, 16, 255 });

        BeginMode3D(camera);

        // Draw Keplerian orbit trails in Solar System mode
        if (g_sys.units == astro::DatasetUnits::SolarSystem) {
            for (size_t i = 1; i < trails.size() && i < g_sys.count; i++) {
                const auto& tr = trails[i];
                if (tr.size() < 2) continue;
                const std::string name = (i < g_sys.names.size()) ? g_sys.names[i] : "";
                Color tc = get_celestial_color(name, i, g_sys.units, 0.0f, 1.0f);
                for (size_t p = 1; p < tr.size(); p++) {
                    float alpha = (float)p / (float)tr.size();
                    DrawLine3D(tr[p - 1], tr[p], ColorAlpha(tc, alpha * 0.55f));
                }
            }
        }

        float max_speed = 5.0f;
        for (size_t i = 0; i < g_sys.count; i += 16) {
            float s = sqrtf(g_sys.vx[i] * g_sys.vx[i] + g_sys.vy[i] * g_sys.vy[i] + g_sys.vz[i] * g_sys.vz[i]);
            if (s > max_speed) max_speed = s;
        }

        // Base particle size dynamically proportioned to scene scale
        float base_sz = max_r * 0.0018f;
        if (base_sz < 0.015f) base_sz = 0.015f;

        for (size_t i = 0; i < g_sys.count; i++) {
            Vector3 pos = { g_sys.x[i], g_sys.y[i], g_sys.z[i] };
            float spd = sqrtf(g_sys.vx[i] * g_sys.vx[i] + g_sys.vy[i] * g_sys.vy[i] + g_sys.vz[i] * g_sys.vz[i]);
            const std::string name = (i < g_sys.names.size()) ? g_sys.names[i] : "";
            Color c = get_celestial_color(name, i, g_sys.units, spd, max_speed);

            float m = g_sys.m[i];

            if (g_sys.count < 150) {
                // Few-body / Solar system: render smooth spheres
                float sz;
                if (i == 0 && m >= 0.5f) {
                    sz = base_sz * 1.0f;
                    DrawSphere(pos, sz, c);
                    DrawSphereWires(pos, sz * 1.15f, 10, 10, (Color){ 255, 200, 50, 160 });
                } else {
                    sz = base_sz * (0.35f + cbrtf(fminf(fmaxf(m, 1e-6f), 1e-2f)) * 1.8f);
                    DrawSphere(pos, sz, c);

                    // Check for Saturn
                    bool is_saturn = false;
                    if (!name.empty()) {
                        std::string lower = name;
                        for (char& ch : lower) ch = (char)tolower((unsigned char)ch);
                        if (lower.find("saturn") != std::string::npos) is_saturn = true;
                    } else if (g_sys.units == astro::DatasetUnits::SolarSystem && i == 7) {
                        is_saturn = true;
                    }

                    if (is_saturn) {
                        // Saturn's tilted rings (26.7 deg axial tilt)
                        const float tilt_rad = 26.7f * DEG2RAD;
                        const Vector3 u1 = { 1.0f, 0.0f, 0.0f };
                        const Vector3 u2 = { 0.0f, cosf(tilt_rad), sinf(tilt_rad) };

                        const int ring_segments = 64;
                        float r_in   = sz * 1.55f;
                        float r_mid1 = sz * 1.95f;
                        float r_mid2 = sz * 2.05f;
                        float r_out  = sz * 2.45f;

                        Color ring_col_b = (Color){ 230, 210, 160, 140 }; // Ring B (dense inner)
                        Color ring_col_a = (Color){ 210, 190, 140, 110 }; // Ring A (outer)

                        for (int s = 0; s < ring_segments; s++) {
                            float a1 = (float)s / ring_segments * 2.0f * PI;
                            float a2 = (float)(s + 1) / ring_segments * 2.0f * PI;
                            float c1 = cosf(a1), s1 = sinf(a1);
                            float c2 = cosf(a2), s2 = sinf(a2);

                            Vector3 d1 = { c1 * u1.x + s1 * u2.x, c1 * u1.y + s1 * u2.y, c1 * u1.z + s1 * u2.z };
                            Vector3 d2 = { c2 * u1.x + s2 * u2.x, c2 * u1.y + s2 * u2.y, c2 * u1.z + s2 * u2.z };

                            // Ring B (inner band)
                            Vector3 b_in1  = { pos.x + d1.x * r_in,   pos.y + d1.y * r_in,   pos.z + d1.z * r_in };
                            Vector3 b_out1 = { pos.x + d1.x * r_mid1, pos.y + d1.y * r_mid1, pos.z + d1.z * r_mid1 };
                            Vector3 b_in2  = { pos.x + d2.x * r_in,   pos.y + d2.y * r_in,   pos.z + d2.z * r_in };
                            Vector3 b_out2 = { pos.x + d2.x * r_mid1, pos.y + d2.y * r_mid1, pos.z + d2.z * r_mid1 };

                            DrawTriangle3D(b_in1, b_out1, b_in2, ring_col_b);
                            DrawTriangle3D(b_out1, b_out2, b_in2, ring_col_b);
                            DrawTriangle3D(b_in1, b_in2, b_out1, ring_col_b);
                            DrawTriangle3D(b_out1, b_in2, b_out2, ring_col_b);

                            // Ring A (outer band)
                            Vector3 a_in1  = { pos.x + d1.x * r_mid2, pos.y + d1.y * r_mid2, pos.z + d1.z * r_mid2 };
                            Vector3 a_out1 = { pos.x + d1.x * r_out,  pos.y + d1.y * r_out,  pos.z + d1.z * r_out };
                            Vector3 a_in2  = { pos.x + d2.x * r_mid2, pos.y + d2.y * r_mid2, pos.z + d2.z * r_mid2 };
                            Vector3 a_out2 = { pos.x + d2.x * r_out,  pos.y + d2.y * r_out,  pos.z + d2.z * r_out };

                            DrawTriangle3D(a_in1, a_out1, a_in2, ring_col_a);
                            DrawTriangle3D(a_out1, a_out2, a_in2, ring_col_a);
                            DrawTriangle3D(a_in1, a_in2, a_out1, ring_col_a);
                            DrawTriangle3D(a_out1, a_in2, a_out2, ring_col_a);

                            // Crisp ring outlines
                            for (float edge_r : { r_in, r_mid1, r_mid2, r_out }) {
                                Vector3 e1 = { pos.x + d1.x * edge_r, pos.y + d1.y * edge_r, pos.z + d1.z * edge_r };
                                Vector3 e2 = { pos.x + d2.x * edge_r, pos.y + d2.y * edge_r, pos.z + d2.z * edge_r };
                                DrawLine3D(e1, e2, (Color){ 245, 230, 180, 180 });
                            }
                        }
                    }
                }
            } else {
                // Dense systems (galaxy disk, star clusters): fast cubes
                if (i == 0 && m > 50.0f) {
                    float sz = base_sz * 2.0f;
                    DrawSphere(pos, sz, (Color){ 255, 240, 160, 255 });
                    DrawSphereWires(pos, sz * 1.2f, 8, 8, (Color){ 255, 200, 50, 180 });
                } else {
                    float sz = base_sz * (0.6f + cbrtf(fminf(m, 10.0f)) * 0.2f);
                    DrawCube(pos, sz, sz, sz, c);
                }
            }
        }

        if (!snapshot_mode && show_octree) {
            draw_octree_wires_recursive(g_backend.get_nodes(), 0);
        }

        EndMode3D();

        // 2D overlays: render name labels, target tracking reticles, and hover highlights
        Vector2 mouse_now = GetMousePosition();
        Vector3 cam_fwd = { camera.target.x - camera.position.x,
                            camera.target.y - camera.position.y,
                            camera.target.z - camera.position.z };

        for (size_t i = 0; i < g_sys.count; i++) {
            Vector3 pos = { g_sys.x[i], g_sys.y[i], g_sys.z[i] };
            Vector3 to_p = { pos.x - camera.position.x, pos.y - camera.position.y, pos.z - camera.position.z };
            if (to_p.x * cam_fwd.x + to_p.y * cam_fwd.y + to_p.z * cam_fwd.z <= 0.0f) continue; // Behind camera!

            Vector2 scr = GetWorldToScreen(pos, camera);
            if (scr.x > 0 && scr.x < screenWidth && scr.y > 0 && scr.y < screenHeight) {
                bool is_tracked = (tracked_idx == static_cast<int>(i));
                float mdist = sqrtf((mouse_now.x - scr.x) * (mouse_now.x - scr.x) + (mouse_now.y - scr.y) * (mouse_now.y - scr.y));
                bool is_hovered = (mdist < 18.0f);

                const std::string name = (i < g_sys.names.size()) ? g_sys.names[i] : "";
                float spd = sqrtf(g_sys.vx[i] * g_sys.vx[i] + g_sys.vy[i] * g_sys.vy[i] + g_sys.vz[i] * g_sys.vz[i]);
                Color body_c = get_celestial_color(name, i, g_sys.units, spd, max_speed);

                // Small crisp beacon point for named / solar bodies so they remain visible at distance
                if (!name.empty() || g_sys.units == astro::DatasetUnits::SolarSystem) {
                    DrawCircleV(scr, (i == 0) ? 3.5f : 2.0f, ColorAlpha(body_c, 0.85f));
                }

                if (is_tracked) {
                    DrawCircleLines((int)scr.x, (int)scr.y, 9, (Color){ 100, 220, 255, 230 });
                    DrawCircleLines((int)scr.x, (int)scr.y, 10, (Color){ 100, 220, 255, 130 });
                } else if (is_hovered) {
                    DrawCircleLines((int)scr.x, (int)scr.y, 8, YELLOW);
                }

                // If body has a name, draw it with its authentic celestial color
                if (!name.empty()) {
                    if (is_tracked) {
                        DrawText(TextFormat("%s [TRACKED]", name.c_str()), (int)scr.x + 12, (int)scr.y - 6, 12, (Color){ 100, 230, 255, 255 });
                    } else if (is_hovered) {
                        DrawText(TextFormat("%s (Click/F to Focus)", name.c_str()), (int)scr.x + 10, (int)scr.y - 6, 12, YELLOW);
                    } else {
                        DrawText(name.c_str(), (int)scr.x + 8, (int)scr.y - 6, 12, ColorAlpha(body_c, 0.95f));
                    }
                }
            }
        }

        int hud_h = (tracked_idx >= 0) ? 190 : ((csv_file != NULL) ? 175 : 155);
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
            DrawText(TextFormat("G: %.6f (%s)", sim_g, (fabsf(sim_g - G_IRL_ASTRO) < 1e-6f) ? "IRL Astro" : "custom"), 30, 106, 14,
                     (fabsf(sim_g - G_IRL_ASTRO) < 1e-6f) ? (Color){100, 220, 120, 255} : LIGHTGRAY);
            if (csv_file) {
                DrawText(TextFormat("Data: %s", GetFileName(csv_file)), 30, 124, 14, LIGHTGRAY);
            }
            if (tracked_idx >= 0 && tracked_idx < static_cast<int>(g_sys.count)) {
                const char* tname = (tracked_idx < static_cast<int>(g_sys.names.size()) && !g_sys.names[tracked_idx].empty())
                                    ? g_sys.names[tracked_idx].c_str() : TextFormat("#%d", tracked_idx);
                DrawText(TextFormat("Target: %s (dist: %.2f)", tname, cam_distance), 30, 142, 14, (Color){ 100, 220, 255, 255 });
                DrawText(TextFormat("Octree (O): %s", show_octree ? "ON" : "OFF"), 30, 160, 14, LIGHTGRAY);
            } else {
                DrawText(TextFormat("Octree (O): %s", show_octree ? "ON" : "OFF"), 30, 142, 14, LIGHTGRAY);
            }
            DrawText("[Click / F] Focus Particle | [C] Center | [TAB] Cycle | [Wheel] Zoom", 30, screenHeight - 68, 12, (Color){ 140, 200, 255, 230 });
            DrawText("[WASD / Right Drag] Pan | [Left Drag] Orbit | [Space] Pause", 30, screenHeight - 50, 12, GRAY);
            DrawText("[Up/Down] Speed | [Left/Right] Step -/+ | [O] Octree", 30, screenHeight - 32, 12, GRAY);
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
