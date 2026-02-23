#!/usr/bin/env python3
"""
Pacman Maze Generator
Generates maze in JSON, PNG, and text formats
"""

import argparse
import sys
from pacman_map import PacmanMap
from maze_renderer import MazeRenderer


def main():
    parser = argparse.ArgumentParser(
        description='Generate Pacman maze in multiple formats',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-o', '--output',
                       default='maze',
                       help='Output filename prefix (default: maze)')
    
    parser.add_argument('-s', '--size',
                       type=int,
                       default=20,
                       help='Tile size in pixels (default: 20)')
    
    parser.add_argument('--seed',
                       type=int,
                       help='Random seed for reproducible mazes')
    
    parser.add_argument('--json-only',
                       action='store_true',
                       help='Generate only JSON file')
    
    parser.add_argument('--png-only',
                       action='store_true',
                       help='Generate only PNG file')
    
    parser.add_argument('--text-only',
                       action='store_true',
                       help='Generate only text file')
    
    parser.add_argument('-v', '--verbose',
                       action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Generate maze
    if args.verbose:
        print("Generating maze...")
    
    maze = PacmanMap(seed=args.seed)
    maze.generate()
    
    if args.verbose:
        print(f"Generated after {maze.gen_count} attempts")
    
    # Determine what to generate
    generate_json = args.json_only or not (args.png_only or args.text_only)
    generate_png = args.png_only or not (args.json_only or args.text_only)
    generate_text = args.text_only or not (args.json_only or args.png_only)
    
    tile_map = maze.get_tile_map()
    
    # Generate JSON
    if generate_json:
        json_file = f"{args.output}.json"
        maze.export_to_graph_json(json_file)
        if args.verbose:
            print(f"JSON: {json_file}")
    
    # Generate PNG
    if generate_png:
        png_file = f"{args.output}.png"
        renderer = MazeRenderer(tile_size=args.size)
        renderer.render_to_file(tile_map, png_file)
        if args.verbose:
            print(f"PNG: {png_file}")
    
    # Generate text
    if generate_text:
        txt_file = f"{args.output}.txt"
        MazeRenderer.render_text_to_file(tile_map, txt_file)
        if args.verbose:
            print(f"Text: {txt_file}")
    
    if not args.verbose:
        print(f"Maze generated: {args.output}.*")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
