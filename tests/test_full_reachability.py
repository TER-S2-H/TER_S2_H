"""
§8 required test: every PATH cell must be reachable from pacman_spawn
via pacman_graph.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.game.maze import Maze


def test_full_reachability(maze):
    """All pellet/energizer cells are reachable from pacman_spawn."""
    reachable = maze.bfs_reachable(maze.pacman_spawn, use_ghost_graph=False)
    pellet_cells = maze.initial_pellets | maze.initial_energizers
    unreachable  = pellet_cells - reachable
    assert not unreachable, (
        f"Unreachable cells from {maze.pacman_spawn}: {sorted(unreachable)}"
    )
