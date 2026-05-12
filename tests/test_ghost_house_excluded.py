"""
§8 required test: no ghost-wall cell appears in pacman_graph's node set.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.game.maze import Maze


def test_ghost_house_excluded(maze):
    """Ghost-house cells (door + interior) are absent from pacman_graph."""
    pac_nodes = set(maze.pacman_graph.keys())
    overlap   = maze.ghost_house & pac_nodes
    assert not overlap, (
        f"Ghost-house cells found in pacman_graph: {sorted(overlap)}"
    )
