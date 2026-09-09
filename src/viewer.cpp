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
    float sim_eps_sq = DEFAULT_EPSILON_SQ;
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
        } else if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) {
            sim_eps_sq = (float)atof(argv[++i]);
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

    if (csv_file && !g_custom) {
        sim_g = G_IRL_ASTRO;
    }

    if (csv_file) {
        cam_distance = 45.0f;
    }

    if (!snapshot_mode) {
        if (!csv_file) {
            g_sys.init_disk(n_bodies, 80.0f, 800.0f, 150.0f);
        }
        g_backend.compute_forces(g_sys, sim_theta, sim_g, sim_eps_sq);
    }

    while (!WindowShouldClose()) {
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
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            cam_distance -= wheel * (cam_distance * 0.08f);
            if (cam_distance < 5.0f) cam_distance = 5.0f;
            if (cam_distance > 5000.0f) cam_distance = 5000.0f;
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
                int substeps = (live_speed >= 2.0f) ? (int)live_speed : 1;
                float dt_step = sim_dt * (live_speed / (float)substeps);
                for (int s = 0; s < substeps; s++) {
                    g_backend.compute_forces(g_sys, sim_theta, sim_g, sim_eps_sq);
                    g_sys.integrate_symplectic(dt_step);
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

        float max_speed = 5.0f;
        for (size_t i = 0; i < g_sys.count; i += 16) {
            float s = sqrtf(g_sys.vx[i]*g_sys.vx[i] + g_sys.vy[i]*g_sys.vy[i] + g_sys.vz[i]*g_sys.vz[i]);
            if (s > max_speed) max_speed = s;
        }

        for (size_t i = 0; i < g_sys.count; i++) {
            Vector3 pos = { g_sys.x[i], g_sys.y[i], g_sys.z[i] };
            if (i == 0 && g_sys.m[0] > 100.0f) {
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

        if (!snapshot_mode && show_octree) {
            draw_octree_wires_recursive(g_backend.get_nodes(), 0);
        }

        EndMode3D();

        int hud_h = (csv_file != NULL) ? 175 : 155;
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
                DrawText(TextFormat("Octree (O): %s", show_octree ? "ON" : "OFF"), 30, 142, 14, LIGHTGRAY);
            } else {
                DrawText(TextFormat("Octree (O): %s", show_octree ? "ON" : "OFF"), 30, 124, 14, LIGHTGRAY);
            }
            DrawText("[Up/Down] Speed | [Space] Pause | [Left/Right] Step -/+ | [O] Octree", 30, screenHeight - 50, 12, GRAY);
            DrawText("[Left Click Drag] Orbit | [Right Click Drag] Pan | [Wheel] Zoom", 30, screenHeight - 32, 12, GRAY);
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
