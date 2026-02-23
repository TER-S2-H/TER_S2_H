#!/usr/bin/env python3
"""
JSON Rendering Test Suite
Tests that JSON export preserves complete maze structure
"""

import sys
import os

# Add parent directory to path
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(test_dir))
sys.path.insert(0, project_root)

from src.pacman_map import PacmanMap
from src.maze_renderer import MazeRenderer
from render_json import JSONMazeRenderer
from PIL import Image
from typing import Tuple


def test_json_rendering(seed: int) -> Tuple[bool, str]:
    """
    Test that JSON rendering matches original for a given seed
    
    Returns:
        (success, message)
    """
    # Generate maze
    maze = PacmanMap(seed=seed)
    maze.generate()
    tile_map = maze.get_tile_map()
    
    # Render original
    renderer = MazeRenderer(tile_size=20)
    original_image = renderer.render_to_image(tile_map)
    
    # Export to JSON using SAME tile_map
    json_data = maze.export_to_graph_json(tile_map=tile_map)
    
    # Render from JSON
    json_renderer = JSONMazeRenderer(tile_size=20)
    json_image = json_renderer.render_to_image(json_data)
    
    # Compare pixel by pixel
    if original_image.size != json_image.size:
        return False, f"Size mismatch: {original_image.size} vs {json_image.size}"
    
    pixels1 = original_image.load()
    pixels2 = json_image.load()
    width, height = original_image.size
    
    different_pixels = 0
    for y in range(height):
        for x in range(width):
            if pixels1[x, y] != pixels2[x, y]:
                different_pixels += 1
    
    if different_pixels == 0:
        return True, "Perfect match"
    else:
        percentage = (different_pixels / (width * height)) * 100
        return False, f"{different_pixels} pixels differ ({percentage:.2f}%)"


def main():
    print("=" * 70)
    print("JSON RENDERING TEST SUITE")
    print("=" * 70)
    
    # Test multiple seeds
    test_seeds = [1, 42, 123, 456, 789, 999, 12345, 67890]
    
    passed = 0
    failed = 0
    
    for seed in test_seeds:
        success, message = test_json_rendering(seed)
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"\nSeed {seed:5d}: {status} - {message}")
        
        if success:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
