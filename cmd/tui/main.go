package main

/*
#cgo CFLAGS: -I../../include -O3 -fopenmp -std=c99 -march=native
#cgo LDFLAGS: -lm -fopenmp
#include "engine.h"
#include <stdlib.h>
*/
import "C"

import (
	"flag"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type tickMsg time.Time

type model struct {
	numBodies int
	preset    string
	paused    bool
	width     int
	height    int
	stats     C.SimStats

	// 2D particle projection cache for terminal rendering
	posX []float32
	posY []float32
}

func initialModel(n int, preset string) model {
	if preset == "three_body" {
		C.sim_init_three_body()
		n = 3
	} else {
		C.sim_init_disk(C.int(n), 100.0, 1000.0, 200.0)
	}

	return model{
		numBodies: n,
		preset:    preset,
		paused:    false,
		width:     80,
		height:    24,
		stats:     C.sim_get_stats(),
		posX:      make([]float32, n),
		posY:      make([]float32, n),
	}
}

func (m model) Init() tea.Cmd {
	return tea.Tick(time.Millisecond*16, func(t time.Time) tea.Msg {
		return tickMsg(t)
	})
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "q", "ctrl+c", "esc":
			return m, tea.Quit
		case " ":
			m.paused = !m.paused
		case "r", "R":
			if m.preset == "three_body" {
				C.sim_init_three_body()
			} else {
				C.sim_init_disk(C.int(m.numBodies), 100.0, 1000.0, 200.0)
			}
			m.stats = C.sim_get_stats()
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

	case tickMsg:
		if !m.paused {
			C.sim_step(0.01, 0.65, 4.0)
			m.stats = C.sim_get_stats()
		}

		// Retrieve particle coordinates for terminal canvas
		if len(m.posX) > 0 {
			C.sim_get_positions_2d((*C.float)(&m.posX[0]), (*C.float)(&m.posY[0]), C.int(len(m.posX)))
		}

		return m, tea.Tick(time.Millisecond*16, func(t time.Time) tea.Msg {
			return tickMsg(t)
		})
	}

	return m, nil
}

func renderCanvas(m model, canvasW, canvasH int) string {
	if canvasW < 10 || canvasH < 5 {
		return ""
	}

	grid := make([][]rune, canvasH)
	for y := 0; y < canvasH; y++ {
		grid[y] = make([]rune, canvasW)
		for x := 0; x < canvasW; x++ {
			grid[y][x] = ' '
		}
	}

	bound := float32(110.0)
	if m.preset == "three_body" {
		bound = 15.0
	}

	halfW := float32(canvasW) / 2.0
	halfH := float32(canvasH) / 2.0

	count := len(m.posX)
	for i := 0; i < count; i++ {
		px := m.posX[i]
		py := m.posY[i]

		col := int(halfW + (px/bound)*halfW)
		row := int(halfH - (py/bound)*halfH*0.55)

		if col >= 0 && col < canvasW && row >= 0 && row < canvasH {
			if i == 0 {
				grid[row][col] = '@' // Supermassive core
			} else {
				current := grid[row][col]
				if current == ' ' {
					grid[row][col] = '.'
				} else if current == '.' {
					grid[row][col] = '*'
				} else if current == '*' {
					grid[row][col] = '#'
				}
			}
		}
	}

	var sb strings.Builder
	for y := 0; y < canvasH; y++ {
		sb.WriteString(string(grid[y]))
		if y < canvasH-1 {
			sb.WriteString("\n")
		}
	}

	return sb.String()
}

func (m model) View() string {
	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("#7D56F4")).
		MarginBottom(1)

	canvasBoxStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#5A56E0")).
		Padding(0, 1)

	statBoxStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#626262")).
		Padding(0, 1).
		Width(34)

	helpStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#626262")).
		MarginTop(1)

	statusBadge := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("#FFFFFF")).
		Background(lipgloss.Color("#04B575")).
		Padding(0, 1).
		Render("RUNNING")

	if m.paused {
		statusBadge = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#FFFFFF")).
			Background(lipgloss.Color("#FF5F87")).
			Padding(0, 1).
			Render("PAUSED")
	}

	// Dynamic sizing
	canvasW := m.width - 42
	if canvasW < 30 {
		canvasW = 30
	}
	if canvasW > 70 {
		canvasW = 70
	}

	canvasH := m.height - 8
	if canvasH < 12 {
		canvasH = 12
	}
	if canvasH > 26 {
		canvasH = 26
	}

	canvasStr := renderCanvas(m, canvasW, canvasH)
	canvasView := canvasBoxStyle.Render(canvasStr)

	energyStr := fmt.Sprintf("%.2e", float64(m.stats.total_energy))
	if math.IsNaN(float64(m.stats.total_energy)) || m.numBodies > 4096 {
		energyStr = "N/A (>4k)"
	}

	statsContent := fmt.Sprintf(
		"State:       %s\n\n"+
			"Preset:      %s\n"+
			"Bodies:      %d\n"+
			"Step:        %d\n"+
			"Octree Tree: %d nodes\n"+
			"Time/Step:   %.2f ms\n"+
			"Engine FPS:  %.1f\n"+
			"Energy:      %s",
		statusBadge,
		m.preset,
		int(m.stats.n),
		int(m.stats.step),
		int(m.stats.active_nodes),
		float64(m.stats.step_time_ms),
		float64(m.stats.fps),
		energyStr,
	)

	statsView := statBoxStyle.Render(statsContent)

	mainView := lipgloss.JoinHorizontal(lipgloss.Top, canvasView, "  ", statsView)

	footer := helpStyle.Render("[Space] Pause/Resume  •  [R] Reset  •  [Q] Quit")

	return lipgloss.JoinVertical(lipgloss.Left,
		titleStyle.Render("✨ Barnes-Hut N-Body Engine (Go + Bubble Tea)"),
		mainView,
		footer,
	)
}

func main() {
	// Minimal options
	numBodies := flag.Int("n", 4096, "Number of bodies in simulation")
	preset := flag.String("preset", "disk", "Preset: 'disk' or 'three_body'")
	flag.Parse()

	p := tea.NewProgram(initialModel(*numBodies, *preset), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Printf("Error running TUI: %v\n", err)
		os.Exit(1)
	}
}
