# Pac-Man Maze Generator

Generate random Pac-Man-style mazes with JSON graph export, PNG visualization, and text rendering.

## Project Structure

```
.
├── generate.py          # Main generator script
├── render_json.py       # Render maze from JSON
├── src/
│   ├── pacman_map.py    # Core maze generation logic
│   └── maze_renderer.py # PNG and text rendering
└── maze_[seed]/         # Generated maze output folders
    ├── maze.json        # Graph representation
    ├── maze.png         # Visual rendering
    └── maze.txt         # ASCII art
```

## Requirements

- Python 3.6 or higher
- Pillow (PIL)

## Installation

```bash
pip install Pillow
```

## Usage

Generate a maze with all three output formats (JSON, PNG, and TXT):

```bash
python3 generate.py
```

This creates a folder named `maze_[seed]` containing:
- `maze.json` - Graph representation with nodes, edges, and walls
- `maze.png` - Visual rendering (560×620 pixels, 20px tiles)
- `maze.txt` - ASCII art representation

### Options

```bash
python3 generate.py [options]

Options:
  -o, --output PATH     Output folder name (default: maze_[seed])
  -s, --size SIZE       Tile size for PNG in pixels (default: 20)
  --seed SEED           Random seed for reproducible mazes
  --json-only           Generate only JSON file
  --png-only            Generate only PNG file
  --text-only           Generate only text file
  -v, --verbose         Print detailed generation info
```

### Examples

```bash
# Generate maze with random seed
python3 generate.py

# Generate with specific seed
python3 generate.py --seed 42

# Generate with custom output folder name
python3 generate.py -o my_maze

# Generate with larger tiles
python3 generate.py -s 30

# Generate only JSON
python3 generate.py --json-only
```

## Rendering from JSON

To verify the maze structure or create visualizations from existing JSON files, use the JSON renderer:

```bash
python3 render_json.py <json_file> [options]
```

This allows you to:
- Verify that JSON contains complete maze structure
- Create images from JSON without regenerating the maze
- Visualize the graph structure with nodes and edges

### Options

```bash
Options:
  -o, --output PATH     Output image file (default: [json_name]_render.png)
  -s, --size SIZE       Tile size in pixels (default: 20)
  --show-graph          Show nodes and edges on the maze
  -v, --verbose         Print detailed information
```

### Examples

```bash
# Render maze from JSON
python3 render_json.py maze_42/maze.json

# Render with custom output and larger tiles
python3 render_json.py maze_42/maze.json -o verification.png -s 30

# Show graph structure (nodes and edges)
python3 render_json.py maze_42/maze.json --show-graph -v
```

## JSON Format

The JSON export represents the maze as a graph:

```json
{
  "width": 28,
  "height": 31,
  "nodes": [
    {"id": "1_1", "x": 1, "y": 1, "type": "PATH"},
    ...
  ],
  "edges": [
    {"from": "1_1", "to": "2_1"},
    ...
  ],
  "walls": [
    {"x": 0, "y": 0},
    ...
  ],
  "ghost_walls": [
    {"x": 12, "y": 14},
    ...
  ],
  "metadata": {
    "node_count": 245,
    "edge_count": 312,
    "wall_count": 423,
    "ghost_wall_count": 2
  }
}
```

## Maze Structure

- 28×31 tile grid
- Ghost house in center with special walls
- Spawn areas (PATHBLANK) outside ghost house
- All walkable tiles included in graph nodes
