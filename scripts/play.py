#!/usr/bin/env python3
"""
Run one Pac-Man game with chosen agents.

Usage:
    python scripts/play.py --maze generated_maze/maze_858558/maze.json \
                           --pacman mcts --ghosts random --record
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents import get_agent
from src.benchmark.run_game import run_game, save_session
from src.game.maze import Maze


def main():
    parser = argparse.ArgumentParser(description="Play one Pac-Man game")
    parser.add_argument("--maze", required=True, help="Path to maze.json")
    parser.add_argument("--pacman", default="mcts_pacman",
                        help="Pac-Man agent name (default: mcts_pacman)")
    parser.add_argument("--ghosts", default="heuristic_ghosts",
                        help="Ghost agent name (default: heuristic_ghosts)")
    parser.add_argument("--record", action="store_true",
                        help="Save session pickle to sessions/")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--out", default=None,
                        help="Output path for session pickle (overrides default)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    maze        = Maze.load(args.maze)
    pac_agent   = get_agent(args.pacman)()
    ghost_agent = get_agent(args.ghosts)()

    print(f"Maze: {args.maze}")
    print(f"Agents: {args.pacman} vs {args.ghosts}")
    print(f"Seed: {args.seed}  Max-steps: {args.max_steps}")

    record = run_game(
        maze, pac_agent, ghost_agent,
        seed      = args.seed,
        max_steps = args.max_steps,
        record    = args.record,
    )

    print(f"\nOutcome:     {record['outcome']}")
    print(f"Total steps: {record['total_steps']}")
    print(f"Pellets remaining: "
          f"{len(record['frames'][-1]['active_pellets']) if record['frames'] else '?'}")

    if args.record:
        os.makedirs("sessions", exist_ok=True)
        # Find next session number
        existing = sorted([
            f for f in os.listdir("sessions") if f.startswith("session_") and f.endswith(".pkl")
        ])
        idx = len(existing)
        out_path = args.out or f"sessions/session_{idx:04d}.pkl"
        save_session(record, out_path)
        print(f"Session saved to {out_path}")


if __name__ == "__main__":
    main()
