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
- `maze.json` - Graph representation
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

## Project Structure

```
.
├── generate.py          - Main maze generator
├── render_json.py       - JSON to image renderer
├── src/
│   ├── pacman_map.py    - Maze generation logic
│   └── maze_renderer.py - Rendering functions
└── maze_[seed]/         - Generated output folders
```

## JSON Format

The maze is exported as a graph structure:

```json
{
  "width": 28,
  "height": 31,
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
    "node_count": 245,
    "edge_count": 312,
    "wall_count": 423,
    "ghost_wall_count": 2
  }
}
```

## License

See LICENSE file for details.
