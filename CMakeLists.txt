cmake_minimum_required(VERSION 3.20)
project(AstroHPC VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Build configuration options
option(ENABLE_CUDA "Enable CUDA GPU acceleration backend" OFF)
option(ENABLE_MPI "Enable distributed MPI parallelism" ON)
option(BUILD_VIEWER "Build Raylib 3D visualizer" ON)
option(BUILD_TUI "Build interactive Go TUI" ON)
option(BUILD_TESTS "Build comprehensive test suite" ON)

# Architecture optimization flag (e.g. -DMARCH=native or -DMARCH=x86-64-v3)
set(MARCH "" CACHE STRING "Architecture optimization flag (e.g. native, x86-64-v3)")
if(MARCH)
    add_compile_options(-march=${MARCH})
endif()

# OpenMP intra-node multi-threading configuration
option(ENABLE_OPENMP "Enable OpenMP multi-threading within nodes" ON)
if(ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP C++ found: ${OpenMP_CXX_VERSION}")
        add_compile_definitions(ASTRO_ENABLE_OPENMP)
    else()
        message(STATUS "OpenMP not found; building single-threaded")
    endif()
endif()

# MPI distributed computing configuration
if(ENABLE_MPI)
    find_package(MPI QUIET)
    if(MPI_CXX_FOUND)
        message(STATUS "MPI C++ found: ${MPI_CXX_COMPILER} (Version: ${MPI_CXX_VERSION})")
        add_compile_definitions(ASTRO_ENABLE_MPI)
    else()
        message(STATUS "MPI not found; building in single-node mode")
    endif()
endif()

# CUDA backend configuration (modular cluster deployment)
if(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(ASTRO_ENABLE_CUDA)
endif()

# Core static engine library
set(CORE_SOURCES
    src/particle_system.cpp
    src/backend/cpu_backend.cpp
    src/backend/backend_factory.cpp
    src/mpi_domain.cpp
    src/snapshot.c
)
if(ENABLE_CUDA)
    list(APPEND CORE_SOURCES src/backend/cuda_backend.cu)
endif()

add_library(astro_core STATIC ${CORE_SOURCES})
target_include_directories(astro_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(astro_core PUBLIC m)
if(OpenMP_CXX_FOUND)
    target_link_libraries(astro_core PUBLIC OpenMP::OpenMP_CXX)
endif()
if(MPI_CXX_FOUND)
    target_link_libraries(astro_core PUBLIC MPI::MPI_CXX)
endif()

# 1. Headless CLI Simulator
add_executable(nbody_sim src/main.cpp)
target_link_libraries(nbody_sim PRIVATE astro_core)

# 2. Raylib 3D Visualizer (astro_view)
if(BUILD_VIEWER)
    find_package(raylib 5.0 QUIET)
    if(NOT raylib_FOUND)
        find_package(PkgConfig QUIET)
        if(PKG_CONFIG_FOUND)
            pkg_check_modules(RAYLIB QUIET raylib)
        endif()
    endif()

    add_executable(astro_view src/viewer.cpp)
    target_link_libraries(astro_view PRIVATE astro_core)

    if(TARGET raylib)
        target_link_libraries(astro_view PRIVATE raylib)
    elseif(RAYLIB_FOUND)
        target_include_directories(astro_view PRIVATE ${RAYLIB_INCLUDE_DIRS})
        target_link_libraries(astro_view PRIVATE ${RAYLIB_LIBRARIES})
    else()
        find_path(RAYLIB_INCLUDE_DIR NAMES raylib.h PATHS /usr/local/include /usr/include)
        find_library(RAYLIB_LIBRARY NAMES raylib PATHS /usr/local/lib /usr/lib /usr/lib64)
        if(RAYLIB_INCLUDE_DIR AND RAYLIB_LIBRARY)
            target_include_directories(astro_view PRIVATE ${RAYLIB_INCLUDE_DIR})
            target_link_libraries(astro_view PRIVATE ${RAYLIB_LIBRARY})
        endif()
    endif()

    # Link platform windowing and graphics libraries
    if(APPLE)
        target_link_libraries(astro_view PRIVATE "-framework OpenGL" "-framework Cocoa" "-framework IOKit" "-framework CoreVideo")
    else()
        target_link_libraries(astro_view PRIVATE GL m pthread dl rt X11)
    endif()
endif()

# 3. Comprehensive HPC Test Suite
if(BUILD_TESTS)
    enable_testing()
    add_executable(test_hpc tests/test_hpc.cpp)
    target_link_libraries(test_hpc PRIVATE astro_core)
    add_test(NAME test_hpc COMMAND test_hpc)
endif()

# 4. Interactive Go TUI (astro_tui)
if(BUILD_TUI)
    find_program(GO_EXECUTABLE go)
    if(GO_EXECUTABLE)
        add_custom_target(astro_tui ALL
            COMMAND ${GO_EXECUTABLE} build -o ${CMAKE_BINARY_DIR}/astro_tui ./cmd/tui
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Building astro_tui via Go..."
        )
    endif()
endif()
