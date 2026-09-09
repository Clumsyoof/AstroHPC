CC = gcc
CFLAGS = -std=c99 -O3 -Wall -Wextra -fopenmp -march=native -Iinclude
LDFLAGS = -lm -fopenmp

SRC = src/main.c src/particles.c src/octree.c
OBJ = $(SRC:.c=.o)
TARGET = nbody_sim
TUI_TARGET = astro_tui

all: $(TARGET) $(TUI_TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

$(TUI_TARGET):
	go build -o $(TUI_TARGET) ./cmd/tui

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET) $(TUI_TARGET)

.PHONY: all clean $(TUI_TARGET)

