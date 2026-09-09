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

# Core Engine Objects (Modern C++ SoA & Backend)
CORE_CPP_SRC = src/particle_system.cpp src/backend/cpu_backend.cpp src/backend/backend_factory.cpp
CORE_CPP_OBJ = $(CORE_CPP_SRC:.cpp=.o)
C_SRC        = src/snapshot.c
C_OBJ        = $(C_SRC:.c=.o)

TARGET = nbody_sim
TUI_TARGET = astro_tui
VIEWER_TARGET = astro_view
TEST_TARGET = test_hpc

VIEWER_OBJ = src/viewer.o $(CORE_CPP_OBJ) $(C_OBJ)
RAYLIB_FLAGS = -L/usr/local/lib -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

all: $(TARGET) $(TUI_TARGET) $(VIEWER_TARGET)

$(TARGET): src/main.o $(CORE_CPP_OBJ) $(C_OBJ) $(CUDA_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(VIEWER_TARGET): $(VIEWER_OBJ)
	$(CXX) $(VIEWER_OBJ) -o $@ $(RAYLIB_FLAGS)

$(TUI_TARGET):
	go build -o $(TUI_TARGET) ./cmd/tui

test: $(TEST_TARGET)
	@./$(TEST_TARGET)

$(TEST_TARGET): tests/test_hpc.o $(CORE_CPP_OBJ) $(C_OBJ) $(CUDA_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f src/*.o src/backend/*.o tests/*.o $(TARGET) $(TUI_TARGET) $(VIEWER_TARGET) $(TEST_TARGET)

.PHONY: all clean test $(TUI_TARGET)
