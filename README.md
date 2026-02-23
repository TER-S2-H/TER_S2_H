# Pac-Man Maze Generator

Generate random Pac-Man-style mazes with JSON graph export, PNG visualization, and text rendering.

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
python3 generate.py -o output
```

This creates:
- `output.json` - Graph representation with nodes, edges, and walls
- `output.png` - Visual rendering (560×620 pixels, 20px tiles)
- `output.txt` - ASCII art representation

### Options

```bash
python3 generate.py -o <output> [options]

Options:
  -o, --output PATH     Output file path (without extension)
  -s, --size SIZE       Tile size for PNG in pixels (default: 20)
  --seed SEED           Random seed for reproducible mazes
  --json-only           Generate only JSON file
  --png-only            Generate only PNG file
  --text-only           Generate only text file
  -v, --verbose         Print detailed generation info
```

### Examples

```bash
# Generate with custom tile size
python3 generate.py -o large_maze -s 30

# Generate only JSON
python3 generate.py -o maze --json-only

# Generate with specific seed
python3 generate.py -o maze --seed 42
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
