CXX ?= g++
CC  ?= gcc

# HPC Architecture Flag: configurable at build time (e.g. make MARCH=native or make MARCH=x86-64-v3)
MARCH ?=
ifneq ($(strip $(MARCH)),)
    ARCH_FLAGS = -march=$(MARCH)
else
    ARCH_FLAGS =
endif

CXXFLAGS = -std=c++17 -O3 -Wall -Wextra $(ARCH_FLAGS) -Iinclude -I/usr/local/include
CFLAGS   = -std=c99 -O3 -Wall -Wextra $(ARCH_FLAGS) -Iinclude -I/usr/local/include
LDFLAGS  = -lm

# Modular CUDA compilation toggle (0 = CPU only, 1 = GPU offload)
ENABLE_CUDA ?= 0
ifeq ($(ENABLE_CUDA), 1)
    NVCC ?= nvcc
    NVCCFLAGS = -std=c++17 -O3 -Iinclude $(ARCH_FLAGS)
    CUDA_SRC = src/backend/cuda_backend.cu
    CUDA_OBJ = src/backend/cuda_backend.o
    CXXFLAGS += -DASTRO_ENABLE_CUDA
    LDFLAGS += -lcudart
endif

# C++ Core Engine
CPP_SRC = src/main.cpp src/particle_system.cpp src/backend/cpu_backend.cpp src/backend/backend_factory.cpp
CPP_OBJ = $(CPP_SRC:.cpp=.o)
C_SRC   = src/snapshot.c
C_OBJ   = $(C_SRC:.c=.o)

TARGET = nbody_sim
TUI_TARGET = astro_tui
VIEWER_TARGET = astro_view

VIEWER_SRC = src/viewer.c src/snapshot.c src/particles.c src/octree.c
VIEWER_OBJ = src/viewer.o src/snapshot.o src/particles.o src/octree.o
RAYLIB_FLAGS = -L/usr/local/lib -lraylib -lGL -lm -lpthread -ldl -lrt -lX11 -fopenmp

TEST_CPP_SRC = tests/test_hpc.cpp src/particle_system.cpp src/backend/cpu_backend.cpp src/backend/backend_factory.cpp
TEST_CPP_OBJ = tests/test_hpc.o src/particle_system.o src/backend/cpu_backend.o src/backend/backend_factory.o
TEST_TARGET = test_hpc

all: $(TARGET) $(TUI_TARGET) $(VIEWER_TARGET)

$(TARGET): $(CPP_OBJ) $(C_OBJ) $(CUDA_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(VIEWER_TARGET): $(VIEWER_OBJ)
	$(CC) $(VIEWER_OBJ) -o $@ $(RAYLIB_FLAGS)

$(TUI_TARGET):
	go build -o $(TUI_TARGET) ./cmd/tui

test: $(TEST_TARGET)
	@./$(TEST_TARGET)

$(TEST_TARGET): $(TEST_CPP_OBJ) $(C_OBJ) $(CUDA_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(CPP_OBJ) $(C_OBJ) $(CUDA_OBJ) $(VIEWER_OBJ) $(TEST_CPP_OBJ) src/octree.o src/particles.o $(TARGET) $(TUI_TARGET) $(VIEWER_TARGET) $(TEST_TARGET)

.PHONY: all clean test $(TUI_TARGET)
