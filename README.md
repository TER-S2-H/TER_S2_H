# Pac-Man Maze Generator

Generate random Pac-Man-style mazes with JSON graph export, PNG visualization, and text rendering.

## Requirements

Python 3.6+ with Pillow library:

```bash
pip install Pillow
```

## Quick Start

Generate a maze with all outputs (JSON, PNG, TXT):

```bash
python3 generate.py
```

Output folder `maze_[seed]/` contains:
- `maze.json` - Graph representation with embedded complexity score
- `maze.png` - Visual rendering (560x620px, 20px tiles)
- `maze.txt` - ASCII art

## Usage

### Generate Maze

```bash
python3 generate.py [options]
```

Options:
- `-o, --output PATH` - Output folder name (default: maze_[seed])
- `--seed SEED` - Random seed for reproducible mazes
- `-s, --size SIZE` - Tile size in pixels (default: 20)
- `--json-only` - Generate only JSON file
- `--png-only` - Generate only PNG file
- `--text-only` - Generate only text file
- `-v, --verbose` - Detailed output

Examples:
```bash
python3 generate.py --seed 42
python3 generate.py -o output -s 30
python3 generate.py --json-only
```

### Render from JSON

```bash
python3 render_json.py <json_file> [options]
```

Options:
- `-o, --output PATH` - Output image file
- `-s, --size SIZE` - Tile size in pixels (default: 20)
- `--show-graph` - Visualize nodes and edges
- `-v, --verbose` - Detailed output

Examples:
```bash
python3 render_json.py maze_42/maze.json
python3 render_json.py maze_42/maze.json -o output.png -s 30
python3 render_json.py maze_42/maze.json --show-graph
```

### Complexity Score

Compute and embed the complexity score in `maze.json`. The score uses a greedy nearest-neighbor algorithm with 2-opt local search optimization to find a near-optimal path that visits all food tiles. The swap iterations are automatically determined based on maze size for optimal results.

**Score Formula:** `score = (food_count / steps) × 10` (capped at 10)

```bash
python3 maze_complexity.py maze_42/maze.json
```

Options:
- `--no-tunnel-wrap` - Disable left-right tunnel wrapping
- `-v, --verbose` - Detailed output

### Complexity Visualization

Create an animated GIF of the greedy path:

```bash
python3 maze_complexity_visualize.py maze_42/maze.json
```

Options:
- `-o, --output PATH` - Output GIF file
- `-s, --size SIZE` - Tile size in pixels (default: 20)
- `--frame-step N` - Steps per frame (default: 5)
- `--max-frames N` - Maximum number of frames (default: 1000)
- `--duration MS` - Frame duration in ms (default: 150)
- `--no-tunnel-wrap` - Disable left-right tunnel wrapping
- `-v, --verbose` - Detailed output

## Project Structure

```
.
├── generate.py                   - Main maze generator (wrapper)
├── maze_complexity.py            - Complexity score (wrapper)
├── maze_complexity_visualize.py  - Path visualization (wrapper)
├── render_json.py                - JSON renderer (wrapper)
├── src/
│   ├── generate.py                   - Generator implementation
│   ├── maze_complexity.py            - Complexity score implementation
│   ├── maze_complexity_visualize.py  - Visualization implementation
│   ├── render_json.py                - JSON renderer implementation
│   ├── pacman_map.py                 - Maze generation logic
│   └── maze_renderer.py              - Rendering functions
└── maze_[seed]/         - Generated output folders
  ├── maze.json        - Graph + complexity score
  ├── maze.png         - Visual rendering
  └── maze.txt         - ASCII art
```

## JSON Format

The maze is exported as a graph structure. Ghost house tiles are excluded from nodes and edges, but `walls` and `ghost_walls` are still included.

```json
{
  "nodes": [
    {"id": "1_1", "x": 1, "y": 1, "type": "PATH"}
  ],
  "edges": [
    {"from": "1_1", "to": "2_1"}
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
    "node_count": 245,
    "edge_count": 312,
    "complexity_score": {
      "start": {"x": 13, "y": 23},
      "food_count": 322,
      "steps": 385,
      "score": 8.36,
      "method": "greedy-nearest-with-2opt",
      "wrap_tunnels": true,
      "width": 28,
      "height": 31,
      "swap_iterations": 3220,
      "improvements_applied": 2
    }
  }
}
```

## License

See LICENSE file for details.
