CC = gcc
CFLAGS = -std=c99 -O3 -Wall -Wextra -fopenmp -march=native -Iinclude
LDFLAGS = -lm -fopenmp

SRC = src/main.c src/particles.c src/octree.c src/snapshot.c
OBJ = $(SRC:.c=.o)
TARGET = nbody_sim
TUI_TARGET = astro_tui
VIEWER_TARGET = astro_view

VIEWER_SRC = src/viewer.c src/snapshot.c src/particles.c src/octree.c
VIEWER_OBJ = src/viewer.o src/snapshot.o src/particles.o src/octree.o
RAYLIB_FLAGS = -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

all: $(TARGET) $(TUI_TARGET) $(VIEWER_TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

$(VIEWER_TARGET): $(VIEWER_OBJ)
	$(CC) $(VIEWER_OBJ) -o $@ $(LDFLAGS) $(RAYLIB_FLAGS)

$(TUI_TARGET):
	go build -o $(TUI_TARGET) ./cmd/tui

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) src/viewer.o $(TARGET) $(TUI_TARGET) $(VIEWER_TARGET)

.PHONY: all clean $(TUI_TARGET)


