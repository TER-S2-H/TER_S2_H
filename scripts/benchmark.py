#!/usr/bin/env python3
"""
Run a benchmark over a directory of mazes.

Usage:
    python scripts/benchmark.py \
        --maze-dir generated_maze \
        --pacman mcts_pacman heuristic_pacman \
        --ghosts random_ghosts heuristic_ghosts \
        --n-games 5 \
        --out sessions/
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents import get_agent
from src.agents import get_agent_names
from src.benchmark.harness import benchmark


def _resolve_agent_names(selected_names: list[str], role: str) -> list[str]:
    if not selected_names or any(name.lower() == "all" for name in selected_names):
        return get_agent_names(role)
    return selected_names


def main():
    parser = argparse.ArgumentParser(description="Benchmark Pac-Man agents")
    parser.add_argument("--maze-dir", default="generated_maze",
                        help="Directory of maze*.json files")
    parser.add_argument("--pacman", nargs="*", default=[],
                        help="Pac-Man agent names. Omit or pass 'all' to benchmark every registered Pac-Man agent.")
    parser.add_argument("--ghosts", nargs="*", default=[],
                        help="Ghost agent names. Omit or pass 'all' to benchmark every registered ghost agent.")
    parser.add_argument("--n-games", type=int, default=5,
                        help="Games per (pacman, ghosts, maze) pairing")
    parser.add_argument("--out", default=None,
                        help="Output directory for sessions + summary.json")
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    pacman_names = _resolve_agent_names(args.pacman, "pacman")
    ghost_names = _resolve_agent_names(args.ghosts, "ghosts")

    logging.info("Pac-Man agents: %s", ", ".join(pacman_names))
    logging.info("Ghost agents: %s", ", ".join(ghost_names))

    pac_factories = {name: get_agent(name) for name in pacman_names}
    gst_factories = {name: get_agent(name) for name in ghost_names}

    report = benchmark(
        maze_dir             = args.maze_dir,
        pacman_agents        = pac_factories,
        ghost_agents         = gst_factories,
        n_games_per_pairing  = args.n_games,
        out_dir              = args.out,
        parallel             = not args.no_parallel,
        max_steps            = args.max_steps,
    )

    print(json.dumps(report.summary(), indent=2))


if __name__ == "__main__":
    main()
