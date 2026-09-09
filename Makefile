CC ?= gcc

# HPC Architecture Flag: configurable at build time (e.g. make MARCH=native or make MARCH=x86-64-v3)
# Default is empty for portable binaries across heterogeneous cluster nodes
MARCH ?=
ifneq ($(strip $(MARCH)),)
    ARCH_FLAGS = -march=$(MARCH)
else
    ARCH_FLAGS =
endif

CFLAGS = -std=c99 -O3 -Wall -Wextra -fopenmp $(ARCH_FLAGS) -Iinclude
LDFLAGS = -lm -fopenmp

SRC = src/main.c src/particles.c src/octree.c src/snapshot.c
OBJ = $(SRC:.c=.o)
TARGET = nbody_sim
TUI_TARGET = astro_tui
VIEWER_TARGET = astro_view

VIEWER_SRC = src/viewer.c src/snapshot.c src/particles.c src/octree.c
VIEWER_OBJ = src/viewer.o src/snapshot.o src/particles.o src/octree.o
RAYLIB_FLAGS = -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

TEST_SRC = tests/test_physics.c src/particles.c src/octree.c src/snapshot.c
TEST_OBJ = tests/test_physics.o src/particles.o src/octree.o src/snapshot.o
TEST_TARGET = test_physics

all: $(TARGET) $(TUI_TARGET) $(VIEWER_TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

$(VIEWER_TARGET): $(VIEWER_OBJ)
	$(CC) $(VIEWER_OBJ) -o $@ $(LDFLAGS) $(RAYLIB_FLAGS)

$(TUI_TARGET):
	go build -o $(TUI_TARGET) ./cmd/tui

test: $(TEST_TARGET)
	@./$(TEST_TARGET)

$(TEST_TARGET): $(TEST_OBJ)
	$(CC) $(TEST_OBJ) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) src/viewer.o tests/test_physics.o $(TARGET) $(TUI_TARGET) $(VIEWER_TARGET) $(TEST_TARGET)

.PHONY: all clean test $(TUI_TARGET)


