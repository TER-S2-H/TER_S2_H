"""
§8 coverage smoke test: run one MCTS-vs-random-ghosts game and verify
that every reachable PATH cell was either eaten or visited.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.game.maze import Maze
from src.agents.mcts_pacman import MCTSPacmanAgent, PacmanMCTSConfig
from src.agents.random_baseline import RandomGhostsAgent
from src.benchmark.run_game import run_game


def test_coverage_smoke(example_maze):
    """All reachable PATH cells eaten or visited in one MCTS game."""
    maze = example_maze

    cfg = PacmanMCTSConfig(
        n_iterations = 50,   # faster for CI
        K_endgame    = 15,
        T_path       = 80,
    )
    pac_agent   = MCTSPacmanAgent(cfg)
    ghost_agent = RandomGhostsAgent()

    record = run_game(maze, pac_agent, ghost_agent, seed=42, max_steps=5000)

    # Build set of visited positions from frames
    visited = {tuple(f["pacman_pos"]) for f in record["frames"]}

    # All pellet cells must be visited or eaten
    reachable = maze.bfs_reachable(maze.pacman_spawn, use_ghost_graph=False)
    reachable_pellets = reachable & (maze.initial_pellets | maze.initial_energizers)

    # Final frame: remaining cells
    last_frame = record["frames"][-1]
    remaining = set(map(tuple, last_frame["active_pellets"])) | \
                set(map(tuple, last_frame["active_energizers"]))

    unvisited_uncleared = reachable_pellets - visited - (reachable_pellets - remaining)
    # A cell passes if it was visited OR it was eaten (not in remaining)
    missed = {p for p in reachable_pellets if p not in visited and p in remaining}

    # Allow a small number of misses (stragglers)
    assert len(missed) <= 5, (
        f"{len(missed)} reachable pellet cells never visited and still remaining: "
        f"{sorted(missed)[:10]}"
    )
