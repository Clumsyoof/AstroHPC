#ifndef CONFIG_H
#define CONFIG_H

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    #include <stdalign.h>
    #define ALIGN64 alignas(64)
#elif defined(__GNUC__) || defined(__clang__)
    #define ALIGN64 __attribute__((aligned(64)))
#else
    #define ALIGN64
#endif

// Static capacity limits
#define MAX_BODIES          16384
#define MAX_OCTREE_NODES    (MAX_BODIES * 8)
#define MAX_OCTREE_DEPTH    32

// Physical and algorithmic defaults
#define G_IRL_ASTRO         0.004300917f // Newton's G in pc * (km/s)^2 / M_sun (1 time unit = 0.9778 Myr)
#define G_SOLAR_SYSTEM      39.4784176f  // Newton's G in AU * (AU/yr)^2 / M_sun (4 * pi^2)
#define DEFAULT_G           1.0f         // Dimensionless N-body standard
#define DEFAULT_DT          0.01f
#define DEFAULT_DT_SOLAR    0.001f       // 0.001 yr (~8.76 hours) for planetary orbits
#define DEFAULT_EPSILON_SQ  0.04f        // Softening parameter (0.2 pc)^2 for star clusters
#define DEFAULT_EPS_SQ_SOLAR 1e-6f       // Softening parameter (0.001 AU)^2 for solar system
#define DEFAULT_THETA       0.65f        // Barnes-Hut Multipole Acceptance Criterion

#endif // CONFIG_H
