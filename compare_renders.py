#!/usr/bin/env python3
"""
Maze Render Comparison Tool
Compares original maze rendering with JSON-based rendering tile-by-tile
"""

import argparse
import sys
from PIL import Image
from typing import Tuple, List
from src.pacman_map import PacmanMap, Tile
from src.maze_renderer import MazeRenderer
from render_json import JSONMazeRenderer


class RenderComparison:
    """Compare two maze renderings"""
    
    def __init__(self, tile_size: int = 20):
        self.tile_size = tile_size
    
    def compare_images_pixel_by_pixel(self, img1: Image.Image, img2: Image.Image) -> Tuple[bool, int]:
        """
        Compare two images pixel by pixel
        
        Returns:
            Tuple of (are_identical, num_different_pixels)
        """
        if img1.size != img2.size:
            return False, -1
        
        pixels1 = img1.load()
        pixels2 = img2.load()
        
        width, height = img1.size
        different_pixels = 0
        
        for y in range(height):
            for x in range(width):
                if pixels1[x, y] != pixels2[x, y]:
                    different_pixels += 1
        
        return different_pixels == 0, different_pixels
    
    def compare_by_tiles(self, img1: Image.Image, img2: Image.Image) -> Tuple[bool, List[Tuple[int, int]]]:
        """
        Compare two images tile by tile
        
        Returns:
            Tuple of (are_identical, list_of_different_tile_coords)
        """
        if img1.size != img2.size:
            return False, []
        
        width, height = img1.size
        tiles_x = width // self.tile_size
        tiles_y = height // self.tile_size
        
        pixels1 = img1.load()
        pixels2 = img2.load()
        
        different_tiles = []
        
        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                # Check if this tile is identical
                tile_matches = True
                for py in range(self.tile_size):
                    for px in range(self.tile_size):
                        x = tile_x * self.tile_size + px
                        y = tile_y * self.tile_size + py
                        if x < width and y < height:
                            if pixels1[x, y] != pixels2[x, y]:
                                tile_matches = False
                                break
                    if not tile_matches:
                        break
                
                if not tile_matches:
                    different_tiles.append((tile_x, tile_y))
        
        return len(different_tiles) == 0, different_tiles
    
    def get_tile_type(self, tile_map: List[List], x: int, y: int) -> str:
        """Get the type of tile at position (x, y)"""
        if x >= len(tile_map) or y >= len(tile_map[0]):
            return "OUT_OF_BOUNDS"
        
        tile = tile_map[x][y]
        state_names = {
            Tile.BLANK: "BLANK",
            Tile.PATH: "PATH",
            Tile.PATHBLANK: "PATHBLANK",
            Tile.WALL: "WALL",
            Tile.GHOSTWALL: "GHOSTWALL",
            Tile.ENERGIZER: "ENERGIZER",
            Tile.GHOSTSPACE: "GHOSTSPACE"
        }
        return state_names.get(tile.state, f"UNKNOWN({tile.state})")


def main():
    parser = argparse.ArgumentParser(
        description='Compare original maze render with JSON-based render',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--seed',
                       type=int,
                       help='Random seed for maze generation')
    
    parser.add_argument('-s', '--size',
                       type=int,
                       default=20,
                       help='Tile size in pixels (default: 20)')
    
    parser.add_argument('--save-images',
                       action='store_true',
                       help='Save comparison images to disk')
    
    parser.add_argument('-v', '--verbose',
                       action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MAZE RENDER COMPARISON TEST")
    print("=" * 70)
    
    # Generate maze
    if args.verbose:
        print(f"\n[1/5] Generating maze with seed {args.seed}...")
    else:
        print(f"\n[1/5] Generating maze...")
    
    maze = PacmanMap(seed=args.seed)
    maze.generate()
    tile_map = maze.get_tile_map()
    
    seed_used = args.seed if args.seed is not None else maze.seed
    print(f"      Seed: {seed_used}, Attempts: {maze.gen_count}")
    
    # Render original (from tile_map)
    print("\n[2/5] Rendering from tile_map...")
    renderer = MazeRenderer(tile_size=args.size)
    original_image = renderer.render_to_image(tile_map)
    
    # Export to JSON (using the SAME tile_map)
    print("\n[3/5] Exporting to JSON...")
    import tempfile
    import json
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_path = f.name
    
    maze.export_to_graph_json(json_path, tile_map=tile_map)
    
    # Load JSON and count elements
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    print(f"      Nodes: {json_data['metadata']['node_count']}")
    print(f"      Edges: {json_data['metadata']['edge_count']}")
    print(f"      Walls: {json_data['metadata']['wall_count']}")
    print(f"      Ghost walls: {json_data['metadata']['ghost_wall_count']}")
    
    # Render from JSON
    print("\n[4/5] Rendering from JSON...")
    json_renderer = JSONMazeRenderer(tile_size=args.size)
    json_maze_data = json_renderer.load_json(json_path)
    json_image = json_renderer.render_to_image(json_maze_data)
    
    # Compare
    print("\n[5/5] Comparing renders...")
    print("\n" + "-" * 70)
    
    comparator = RenderComparison(tile_size=args.size)
    
    # Pixel-by-pixel comparison
    pixel_match, diff_pixels = comparator.compare_images_pixel_by_pixel(original_image, json_image)
    
    total_pixels = original_image.size[0] * original_image.size[1]
    print(f"\nPIXEL-BY-PIXEL COMPARISON:")
    print(f"  Image size: {original_image.size[0]}x{original_image.size[1]} = {total_pixels:,} pixels")
    print(f"  Different pixels: {diff_pixels:,}")
    
    if pixel_match:
        print(f"  Result: ✓ PERFECT MATCH")
    else:
        percentage = (diff_pixels / total_pixels) * 100
        print(f"  Result: ✗ MISMATCH ({percentage:.2f}% different)")
    
    # Tile-by-tile comparison
    tile_match, diff_tiles = comparator.compare_by_tiles(original_image, json_image)
    
    tiles_x = original_image.size[0] // args.size
    tiles_y = original_image.size[1] // args.size
    total_tiles = tiles_x * tiles_y
    
    print(f"\nTILE-BY-TILE COMPARISON:")
    print(f"  Grid size: {tiles_x}x{tiles_y} = {total_tiles} tiles")
    print(f"  Different tiles: {len(diff_tiles)}")
    
    if tile_match:
        print(f"  Result: ✓ PERFECT MATCH")
    else:
        percentage = (len(diff_tiles) / total_tiles) * 100
        print(f"  Result: ✗ MISMATCH ({percentage:.2f}% different)")
        
        if args.verbose and len(diff_tiles) <= 20:
            print(f"\n  Different tile coordinates:")
            for tx, ty in diff_tiles[:20]:
                tile_type = comparator.get_tile_type(tile_map, tx, ty)
                print(f"    - ({tx:2d}, {ty:2d}) = {tile_type}")
            if len(diff_tiles) > 20:
                print(f"    ... and {len(diff_tiles) - 20} more")
    
    print("-" * 70)
    
    # Save images if requested
    if args.save_images:
        print("\n[EXTRA] Saving images...")
        original_image.save(f"comparison_original_{seed_used}.png")
        json_image.save(f"comparison_json_{seed_used}.png")
        print(f"  Saved: comparison_original_{seed_used}.png")
        print(f"  Saved: comparison_json_{seed_used}.png")
    
    # Clean up temp JSON
    import os
    os.unlink(json_path)
    
    # Final verdict
    print("\n" + "=" * 70)
    if pixel_match and tile_match:
        print("VERDICT: ✓ JSON export and rendering are IDENTICAL")
        print("=" * 70)
        return 0
    else:
        print("VERDICT: ✗ JSON export and rendering have DIFFERENCES")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
