"""ICE gUCT should respect the per-tick compute budget."""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.iceguct_ghosts import IceGuctGhosts
from src.game.types import Direction, Observation


def test_iceguct_compute_budget(example_maze):
    maze = example_maze
    agent = IceGuctGhosts()
    agent.reset(maze, "ghosts")

    junction = next(iter(agent._topology.junctions))
    obs = Observation(
        step=0,
        pacman_pos=maze.pacman_spawn,
        ghost_positions=[junction, junction, junction, junction],
        ghost_frightened=[0, 0, 0, 0],
        active_pellets=list(maze.initial_pellets),
        active_energizers=list(maze.initial_energizers),
    )

    budget_seconds = agent.cfg.budget_ms_per_tick / 1000.0 * 1.2
    for _ in range(100):
        start = time.perf_counter()
        actions = agent.act(obs)
        elapsed = time.perf_counter() - start
        assert elapsed <= budget_seconds
        assert len(actions) == 4
