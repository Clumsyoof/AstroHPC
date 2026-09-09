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
#define DEFAULT_G           1.0f
#define DEFAULT_DT          0.01f
#define DEFAULT_EPSILON_SQ  4.0f    // Gravitational softening parameter
#define DEFAULT_THETA       0.65f   // Barnes-Hut Multipole Acceptance Criterion

#endif // CONFIG_H
