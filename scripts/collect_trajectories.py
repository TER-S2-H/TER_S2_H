#!/usr/bin/env python3
"""Collect Pac-Man trajectories for the ICE gUCT KNN predictor."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents import get_agent
from src.agents.iceguct_ghosts import FeatureExtractor
from src.benchmark.run_game import run_game, save_session
from src.game.maze import Maze
from src.game.types import Observation


logger = logging.getLogger(__name__)


def _maze_paths(maze_dir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(maze_dir, "**", "maze.json"), recursive=True))
    if not paths:
        paths = sorted(glob.glob(os.path.join(maze_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(f"No maze*.json files found under {maze_dir}")
    return paths


def _alias_agent_name(name: str, *, role: str) -> str:
    aliases = {
        "heuristic": f"heuristic_{role}",
        "random": f"random_{role}",
        "mcts": f"mcts_{role}",
    }
    return aliases.get(name, name)


def _extract_pairs(record: dict, maze: Maze) -> Tuple[np.ndarray, np.ndarray]:
    extractor = FeatureExtractor(maze)
    frames = record.get("frames", [])
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for before_raw, after_raw in zip(frames, frames[1:]):
        before = Observation.from_dict(before_raw)
        after = Observation.from_dict(after_raw)
        if before.pacman_pos == after.pacman_pos:
            continue
        xs.append(extractor.feature_vector(before))
        ys.append(extractor.move_vector(before, after))

    if not xs:
        return np.empty((0, 10), dtype=np.float32), np.empty((0, 10), dtype=np.float32)
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def _load_existing_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.empty((0, 10), dtype=np.float32), np.empty((0, 10), dtype=np.float32)
    with np.load(path) as data:
        x = np.asarray(data.get("X", np.empty((0, 10))), dtype=np.float32)
        y = np.asarray(data.get("Y", np.empty((0, 10))), dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError(f"Existing dataset has invalid shapes: X={x.shape}, Y={y.shape}")
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect trajectory data for the ICE gUCT KNN predictor")
    parser.add_argument("--maze-dir", default="generated_maze", help="Directory containing maze*.json files")
    parser.add_argument("--pacman", default="heuristic", help="Pac-Man agent name or alias")
    parser.add_argument("--ghosts", default="random", help="Ghost agent name or alias")
    parser.add_argument("--n-games", type=int, default=100, help="Number of games to collect")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--out", required=True, help="Output directory for session pickles and dataset.npz")
    parser.add_argument("--max-steps", type=int, default=5000, help="Maximum number of simulator steps per game")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pacman_name = _alias_agent_name(args.pacman, role="pacman")
    ghost_name = _alias_agent_name(args.ghosts, role="ghosts")
    pac_factory = get_agent(pacman_name)
    ghost_factory = get_agent(ghost_name)

    maze_paths = _maze_paths(args.maze_dir)
    all_x, all_y = _load_existing_dataset(out_dir / "dataset.npz")
    outcomes: dict[str, int] = {"eaten": 0, "cleared": 0, "timeout": 0}
    total_pairs = 0

    for game_idx in range(args.n_games):
        maze_path = maze_paths[game_idx % len(maze_paths)]
        maze = Maze.load(maze_path)
        pac_agent = pac_factory()
        ghost_agent = ghost_factory()
        seed = args.seed + game_idx
        record = run_game(
            maze,
            pac_agent,
            ghost_agent,
            seed=seed,
            max_steps=args.max_steps,
            record=True,
        )
        save_session(record, str(out_dir / f"session_{game_idx:04d}.pkl"))
        outcomes[record["outcome"]] = outcomes.get(record["outcome"], 0) + 1

        x, y = _extract_pairs(record, maze)
        if x.size:
            all_x = np.concatenate([all_x, x], axis=0) if all_x.size else x
            all_y = np.concatenate([all_y, y], axis=0) if all_y.size else y
            total_pairs += len(x)

    np.savez(out_dir / "dataset.npz", X=all_x, Y=all_y)

    print(json.dumps({
        "games": args.n_games,
        "outcomes": outcomes,
        "pairs_collected": int(total_pairs),
        "dataset_size": int(len(all_x)),
        "output_dir": str(out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
