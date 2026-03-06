# Pac-Man Maze Generator & Game

Generate random Pac-Man-style mazes, score them, animate the optimal path, and play them interactively.

## Requirements

Python 3.6+ with:

```bash
pip install Pillow          # maze generation, rendering, GIF output
pip install pygame          # interactive game only
```

## Quick Start

```bash
# 1. Generate a maze (JSON + PNG + TXT — maze structure only, no pellets)
python3 generate.py --seed 42

# 2. Score it (embeds complexity score in maze.json)
python3 maze_complexity.py maze_42/maze.json -v

# 3. Animate the best-found path
python3 maze_complexity_visualize.py maze_42/maze.json -v

# 4. Play it!
python3 pacman_game.py maze_42/maze.json
```

Output folder `maze_[seed]/` contains:
- `maze.json` — Graph + embedded complexity score (maze structure only)
- `maze.png` — Visual rendering (560×620 px at 20 px/tile, maze structure only)
- `maze.txt` — ASCII art (maze structure only)
- `maze_path.gif` — Animated optimal path (after running the visualizer)

---

## Usage

### Generate a Maze

```bash
python3 generate.py [options]
```

| Option | Default | Description |
|---|---|---|
| `-o, --output PATH` | `maze_[seed]` | Output folder name |
| `--seed SEED` | random | Reproducible seed |
| `-s, --size SIZE` | `20` | Tile size in pixels |
| `--json-only` | — | Skip PNG and TXT |
| `--png-only` | — | Skip JSON and TXT |
| `--text-only` | — | Skip JSON and PNG |
| `-v, --verbose` | — | Verbose output |

```bash
python3 generate.py --seed 42
python3 generate.py -o my_maze -s 30
python3 generate.py --json-only
```

---

### Render from JSON

Re-render a maze image from an existing `maze.json`:

```bash
python3 render_json.py <maze.json> [options]
```

| Option | Default | Description |
|---|---|---|
| `-o, --output PATH` | auto | Output image path |
| `-s, --size SIZE` | `20` | Tile size in pixels |
| `--show-graph` | — | Overlay nodes and edges |
| `-v, --verbose` | — | Verbose output |

```bash
python3 render_json.py maze_42/maze.json
python3 render_json.py maze_42/maze.json -o out.png -s 30 --show-graph
```

---

### Play the Game

Loads a `maze.json` and launches an interactive Pac-Man game.
Pellets, energizers, and Pac-Man are **not** stored in the JSON — they are
computed at runtime from the maze graph.

```bash
python3 pacman_game.py <maze.json> [options]
```

| Option | Default | Description |
|---|---|---|
| `-s, --size SIZE` | `20` | Tile size in pixels |

```bash
python3 pacman_game.py maze_42/maze.json
python3 pacman_game.py maze_42/maze.json -s 24
```

**Controls:**

| Key | Action |
|---|---|
| Arrow keys | Move Pac-Man |
| `R` | Restart |
| `Q` / `Escape` | Quit |

**Gameplay features:**
- **4 ghosts** — Blinky (red), Pinky (pink), Inky (cyan), Clyde (orange) — each with
  classic targeting behaviour from the original arcade game.
- **Ghost modes** cycle automatically: scatter → chase → scatter → chase (repeating).
  Eating a power pellet triggers **frightened** mode (ghosts flash before returning).
- **Ghost house:** ghosts start inside, patrol left/right, then exit via the house door
  one by one on a staggered release schedule (Blinky instant, +3 s each after).
- **Lives system** (3 lives). Touched by a ghost → respawn; lose all lives → Game Over.
- **Score:** dots = 10 pts, energizers = 50 pts, eating a frightened ghost = 200/400/800/1600 pts.
- **Win condition:** clear all dots and energizers.

**Pellet placement (auto-computed at runtime, not stored in `maze.json`):**
- Every walkable `PATH` tile gets a pellet **except**:
  - Ghost-house surround zone: `x ∈ [9, 18]`, `y ∈ [10, 16]`
  - Tunnel edge columns: `x == 0` or `x == 27`
  - Pac-Man spawn tile: `(13, 23)`
- **4 energizers** placed near corner anchors `(1, 3)`, `(26, 3)`, `(1, 23)`, `(26, 23)`,
  snapped to the nearest eligible tile.

**Architecture:**
- `GameState` — all mutable game data: maze graph, Pac-Man, 4 ghost AI agents, pellets,
  score, lives, scatter/chase mode-cycle timer.
- `GameRenderer` — pure drawing (no game logic): maze walls, pellets, energizers, Pac-Man
  with mouth animation, ghost bodies with eyes (full colour / frightened flash / dead silhouette).
- `GhostState` — per-ghost state machine (`waiting → exiting → scatter/chase/frightened/dead`).

---

Computes the minimum steps to visit all food tiles from a fixed start position
(open walk — no return required) and embeds the result in `maze.json`.

**Problem:** Metric TSP Hamiltonian Path — NP-hard in general.

**Algorithm:**
- **≤ 20 nodes:** Held-Karp exact DP — provably optimal, O(2ⁿ · n²).
- **> 20 nodes:** Multi-start local search (`ms-or3opt`):
  1. Several greedy nearest-neighbour starts (first deterministic, rest with randomised top-3 tie-breaking).
  2. Each start is refined through a full pipeline:
     - Deterministic 2-opt (all pairs, repeat passes until convergence)
     - Or-opt for chain lengths 1, 2, 3 (relocate + reverse, best-improvement)
     - Second 2-opt pass (Or-opt can open new 2-opt gains)
     - Random 3-opt segment-swap moves (escape local optima)
     - Final 2-opt pass
  3. The best result across all starts is kept.

**Lower bound:** MST weight ≤ true optimum ≤ best found.
The gap `best_found − mst_lower_bound` indicates how far from optimal the solution is.

**Score formula:** `score = (food_count / steps) × 10` (capped at 10)

```bash
python3 maze_complexity.py <maze.json> [options]
```

| Option | Description |
|---|---|
| `--no-tunnel-wrap` | Disable left-right tunnel wrapping |
| `--held-karp` | Force Held-Karp exact DP (requires ≤ 20 nodes) |
| `-v, --verbose` | Print score, steps, food count, MST lower bound, method |

```bash
python3 maze_complexity.py maze_42/maze.json -v
# Complexity score: 8.63 (steps: 373, food: 322, mst_lb: 321, method: ms-or3opt)
```

---

### Complexity Visualization

Generates an animated GIF showing the best-found path walking through all food tiles:

```bash
python3 maze_complexity_visualize.py <maze.json> [options]
```

| Option | Default | Description |
|---|---|---|
| `-o, --output PATH` | `maze_path.gif` | Output GIF path |
| `-s, --size SIZE` | `20` | Tile size in pixels |
| `--frame-step N` | `5` | Steps per animation frame |
| `--max-frames N` | `1000` | Maximum frames in GIF |
| `--duration MS` | `150` | Frame duration in milliseconds |
| `--no-tunnel-wrap` | — | Disable tunnel wrapping |
| `-v, --verbose` | — | Print steps, method, MST lower bound |

```bash
python3 maze_complexity_visualize.py maze_42/maze.json -v
python3 maze_complexity_visualize.py maze_42/maze.json -o path.gif --frame-step 3
```

---

## Project Structure

```
.
├── generate.py                   — Wrapper: maze generator
├── maze_complexity.py            — Wrapper: complexity scorer
├── maze_complexity_visualize.py  — Wrapper: path animator
├── render_json.py                — Wrapper: JSON renderer
├── pacman_game.py                — Wrapper: interactive game
└── src/
    ├── generate.py               — Generator implementation
    ├── maze_complexity.py        — Complexity scoring (TSP solver)
    ├── maze_complexity_visualize.py — GIF animation
    ├── render_json.py            — Image renderer
    ├── pacman_map.py             — Maze generation logic
    ├── maze_renderer.py          — Rendering helpers
    └── pacman_game.py            — Game logic + pygame renderer

maze_[seed]/                      — Generated output folder
├── maze.json                     — Graph + complexity score (maze only)
├── maze.png                      — Visual rendering (maze only)
├── maze.txt                      — ASCII art (maze only)
└── maze_path.gif                 — Animated path (optional)
```

> **Note:** `maze.json`, `maze.png`, and `maze.txt` contain only the maze
> structure (walls, paths, graph). Pellets, energizers, and Pac-Man are
> computed at runtime by `pacman_game.py` and never written to disk.

All root-level scripts are thin wrappers that import from `src/` and call `main()`.

## JSON Format

Ghost-house tiles are excluded from `nodes` and `edges` but represented in `ghost_walls`.
Left-right tunnel edges are added at runtime when `wrap_tunnels` is enabled.

```json
{
  "nodes": [
    {"id": "1,1", "x": 1, "y": 1, "type": "PATH"}
  ],
  "edges": [
    {"from": "1,1", "to": "2,1"}
  ],
  "walls": [
    {"x": 0, "y": 0}
  ],
  "ghost_walls": [
    {"x": 12, "y": 14}
  ],
  "metadata": {
    "width": 28,
    "height": 31,
    "node_count": 322,
    "edge_count": 418,
    "complexity_score": {
      "start": {"x": 13, "y": 23},
      "food_count": 322,
      "steps": 373,
      "score": 8.63,
      "method": "ms-or3opt",
      "mst_lower_bound": 321,
      "wrap_tunnels": true,
      "width": 28,
      "height": 31
    }
  }
}
```

### `complexity_score` fields

| Field | Description |
|---|---|
| `start` | Fixed start position `(13, 23)` |
| `food_count` | Number of reachable food tiles |
| `steps` | Best-found open-walk length (sum of BFS distances) |
| `score` | `(food_count / steps) × 10`, capped at 10 |
| `method` | `ms-or3opt` (large mazes) or `held-karp-exact` (≤ 20 nodes) |
| `mst_lower_bound` | MST weight — guaranteed lower bound on the true optimum |
| `wrap_tunnels` | Whether left-right tunnels were included in BFS |

---

## License

See [LICENSE](LICENSE) for details.
