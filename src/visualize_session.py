#!/usr/bin/env python3
"""Visualize one Pac-Man simulator session over a maze using pygame.

Usage:
    python src/visualize_session.py generated_maze/maze_858558/maze.json sessions_test/session_0000.pkl
"""

import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple

import pygame

Color = Tuple[int, int, int]


def load_maze(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def load_session(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_paths(args: argparse.Namespace) -> Tuple[str, str]:
    if args.seed is not None and args.session is not None:
        seed = str(args.seed)
        session_idx = int(args.session)
        maze_json = os.path.join(args.maze_root, f"maze_{seed}", "maze.json")
        session_pkl = os.path.join(args.sessions_root, seed, f"session_{session_idx:04d}.pkl")
        return maze_json, session_pkl

    if args.maze_json and args.session_pkl:
        return args.maze_json, args.session_pkl

    raise ValueError(
        "Provide either positional arguments <maze_json> <session_pkl> "
        "or both --seed and --session."
    )


def draw_static_maze(screen: pygame.Surface, maze: Dict, tile_size: int) -> None:
    wall_color: Color = (33, 33, 222)
    ghost_wall_color: Color = (255, 255, 255)

    for wall in maze.get("walls", []):
        x = wall["x"] * tile_size
        y = wall["y"] * tile_size
        pygame.draw.rect(screen, wall_color, pygame.Rect(x, y, tile_size, tile_size))

    for gate in maze.get("ghost_walls", []):
        x = gate["x"] * tile_size
        y = gate["y"] * tile_size
        line_y = y + tile_size // 2
        pygame.draw.line(screen, ghost_wall_color, (x, line_y), (x + tile_size - 1, line_y), 3)


def draw_frame(
    screen: pygame.Surface,
    frame: Dict,
    maze: Dict,
    tile_size: int,
) -> None:
    bg_color: Color = (0, 0, 0)
    pellet_color: Color = (245, 245, 245)
    energizer_color: Color = (255, 184, 82)
    pacman_color: Color = (255, 221, 0)

    ghost_colors: List[Color] = [
        (255, 0, 0),      # red
        (255, 105, 180),  # pink
        (0, 255, 255),    # cyan
        (255, 165, 0),    # orange
    ]
    frightened_color: Color = (70, 120, 255)

    screen.fill(bg_color)

    draw_static_maze(screen, maze, tile_size)

    # Pellets
    pellet_radius = max(2, tile_size // 7)
    for p in frame.get("active_pellets", []):
        cx = p[0] * tile_size + tile_size // 2
        cy = p[1] * tile_size + tile_size // 2
        pygame.draw.circle(screen, pellet_color, (cx, cy), pellet_radius)

    # Energizers
    energizer_radius = max(3, tile_size // 4)
    for e in frame.get("active_energizers", []):
        cx = e[0] * tile_size + tile_size // 2
        cy = e[1] * tile_size + tile_size // 2
        pygame.draw.circle(screen, energizer_color, (cx, cy), energizer_radius)

    # Ghosts
    ghost_positions = frame.get("ghost_positions", [])
    ghost_frightened = frame.get("ghost_frightened", [0, 0, 0, 0])
    ghost_radius = max(5, tile_size // 2 - 1)

    for i, pos in enumerate(ghost_positions):
        cx = pos[0] * tile_size + tile_size // 2
        cy = pos[1] * tile_size + tile_size // 2
        color = frightened_color if i < len(ghost_frightened) and ghost_frightened[i] > 0 else ghost_colors[i % len(ghost_colors)]
        pygame.draw.circle(screen, color, (cx, cy), ghost_radius)

    # Pac-Man (draw last so it stays visible)
    pac = frame.get("pacman_pos", (0, 0))
    pac_radius = max(5, tile_size // 2 - 1)
    pac_cx = pac[0] * tile_size + tile_size // 2
    pac_cy = pac[1] * tile_size + tile_size // 2
    pygame.draw.circle(screen, pacman_color, (pac_cx, pac_cy), pac_radius)

def run_viewer(maze: Dict, session: Dict, tile_size: int, fps: int, loop: bool) -> None:
    frames = session.get("frames", [])
    if not frames:
        raise ValueError("Session contains no frames")

    width = maze["metadata"]["width"]
    height = maze["metadata"]["height"]

    pygame.init()
    pygame.display.set_caption("Pac-Man Session Visualizer")

    hud_height = 56
    screen = pygame.display.set_mode((width * tile_size, height * tile_size + hud_height))
    font = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()

    frame_idx = 0
    paused = False
    running = True

    while running:
        dt = clock.tick(fps)
        _ = dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    frame_idx = min(frame_idx + 1, len(frames) - 1)
                    paused = True
                elif event.key == pygame.K_LEFT:
                    frame_idx = max(frame_idx - 1, 0)
                    paused = True
                elif event.key == pygame.K_HOME:
                    frame_idx = 0
                    paused = True
                elif event.key == pygame.K_END:
                    frame_idx = len(frames) - 1
                    paused = True

        # Draw scene on a subsurface shifted for HUD
        maze_surface = screen.subsurface(pygame.Rect(0, hud_height, width * tile_size, height * tile_size))
        draw_frame(
            maze_surface,
            frames[frame_idx],
            maze,
            tile_size,
        )

        # Clear and redraw HUD area after frame draw.
        pygame.draw.rect(screen, (12, 12, 12), pygame.Rect(0, 0, width * tile_size, hud_height))

        hud_text = (
            "Controls: SPACE play/pause | LEFT/RIGHT step | HOME start | END last | Q/ESC quit"
        )
        controls = font.render(hud_text, True, (200, 200, 200))
        screen.blit(controls, (8, 36))

        # Re-draw the informational lines directly in the top area.
        info1 = font.render(
            f"Session outcome: {session.get('outcome', 'unknown')} | total_steps: {session.get('total_steps', 'n/a')} | {'PAUSED' if paused else 'PLAYING'}",
            True,
            (255, 255, 255),
        )
        info2 = font.render(
            f"Frame {frame_idx}/{len(frames) - 1} | Step {frames[frame_idx].get('step', frame_idx)} | "
            f"Pellets {len(frames[frame_idx].get('active_pellets', []))} | "
            f"Energizers {len(frames[frame_idx].get('active_energizers', []))}",
            True,
            (255, 255, 255),
        )
        screen.blit(info1, (8, 8))
        screen.blit(info2, (8, 22))

        pygame.display.flip()

        if not paused:
            if frame_idx < len(frames) - 1:
                frame_idx += 1
            elif loop:
                frame_idx = 0
            else:
                paused = True

    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Pac-Man session playback")
    parser.add_argument("maze_json", nargs="?", help="Path to maze JSON")
    parser.add_argument("session_pkl", nargs="?", help="Path to session pickle")
    parser.add_argument("--seed", help="Maze seed (uses <maze_root>/maze_<seed>/maze.json)")
    parser.add_argument("--session", help="Session index, e.g. 0000")
    parser.add_argument("--maze-root", default="generated_maze", help="Root folder containing maze_<seed>/maze.json")
    parser.add_argument("--sessions-root", default="sessions", help="Root folder containing per-seed session folders")
    parser.add_argument("-s", "--tile-size", type=int, default=22, help="Tile size in pixels")
    parser.add_argument("--fps", type=int, default=12, help="Playback speed (frames per second)")
    parser.add_argument("--loop", action="store_true", help="Loop playback when reaching the end")
    args = parser.parse_args()

    maze_json, session_pkl = resolve_paths(args)

    if not os.path.exists(maze_json):
        raise FileNotFoundError(f"Maze file not found: {maze_json}")
    if not os.path.exists(session_pkl):
        raise FileNotFoundError(f"Session file not found: {session_pkl}")

    maze = load_maze(maze_json)
    session = load_session(session_pkl)

    run_viewer(maze, session, tile_size=args.tile_size, fps=args.fps, loop=args.loop)


if __name__ == "__main__":
    main()
