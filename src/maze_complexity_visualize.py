#!/usr/bin/env python3
"""
Visualize the greedy food-search path over a maze as an animated GIF.
"""

import argparse
import json
import os
from typing import List, Tuple

from PIL import Image, ImageDraw

from .maze_complexity import compute_greedy_path
from .render_json import JSONMazeRenderer

Position = Tuple[int, int]


def _to_pixel(position: Position, tile_size: int) -> Tuple[int, int]:
    x, y = position
    return x * tile_size + tile_size // 2, y * tile_size + tile_size // 2


def _draw_frame(base: Image.Image, points: List[Tuple[int, int]], current: Tuple[int, int], tile_size: int) -> Image.Image:
    image = base.copy()
    draw = ImageDraw.Draw(image)

    if len(points) > 1:
        draw.line(points, fill=(255, 0, 0), width=max(1, tile_size // 6))

    radius = max(2, tile_size // 3)
    cx, cy = current
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=(255, 255, 0),
        outline=(255, 255, 255),
    )

    return image


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize greedy FoodSearch path as an animated GIF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("maze_json", help="Path to maze.json")
    parser.add_argument("-o", "--output", help="Output GIF file")
    parser.add_argument("-s", "--size", type=int, default=20, help="Tile size in pixels")
    parser.add_argument("--frame-step", type=int, default=5, help="Steps per frame")
    parser.add_argument("--max-frames", type=int, default=1000, help="Maximum number of frames")
    parser.add_argument("--duration", type=int, default=150, help="Frame duration in ms")
    parser.add_argument("--no-tunnel-wrap", action="store_true", help="Disable left-right tunnel wrapping")
    parser.add_argument(
        "--swap-iterations",
        type=int,
        default=0,
        help="Try this many random swaps to shorten the greedy route",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    wrap_tunnels = not args.no_tunnel_wrap

    with open(args.maze_json, "r") as f:
        maze_data = json.load(f)

    steps, path, debug = compute_greedy_path(
        maze_data,
        start=(13, 23),
        wrap_tunnels=wrap_tunnels,
        swap_iterations=args.swap_iterations,
    )

    if args.verbose:
        print("Path steps: %s" % steps)
        print("Food count: %s" % debug.get("food_count"))
        print("Swap iterations: %s" % debug.get("swap_iterations"))
        print("Swaps applied: %s" % debug.get("swaps_applied"))

    renderer = JSONMazeRenderer(tile_size=args.size)
    base_image = renderer.render_to_image(maze_data, show_graph=False)

    positions_px = [_to_pixel(pos, args.size) for pos in path]

    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.maze_json)[0]
        output_path = base_name + "_path.gif"

    step = max(1, args.frame_step)
    frame_indices = list(range(0, len(positions_px), step))
    if len(frame_indices) > args.max_frames:
        step = max(1, len(positions_px) // args.max_frames)
        frame_indices = list(range(0, len(positions_px), step))

    frames: List[Image.Image] = []
    for idx in frame_indices:
        points = positions_px[: idx + 1]
        current = positions_px[idx]
        frame = _draw_frame(base_image, points, current, args.size)
        frames.append(frame)

    if not frames:
        raise ValueError("No frames generated for visualization")

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=0,
    )

    if args.verbose:
        print("GIF saved to: %s" % os.path.abspath(output_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
