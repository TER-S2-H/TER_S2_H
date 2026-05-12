"""Cold-start KNN predictor should warn and fall back to random legal moves."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.iceguct_ghosts import IceGuctGhosts
from src.game.simulator import initial_state
from src.game.types import Observation


def test_knn_cold_start_warns_and_falls_back(example_maze, tmp_path):
    missing = tmp_path / "missing-dataset.npz"

    with pytest.warns(RuntimeWarning):
        agent = IceGuctGhosts(knn_dataset_path=missing, rng=np.random.default_rng(0))
        agent.reset(example_maze, "ghosts")

    state = initial_state(example_maze)
    obs = Observation(
        step=state.step,
        pacman_pos=state.pacman_pos,
        ghost_positions=state.ghost_positions,
        ghost_frightened=state.ghost_frightened,
        active_pellets=list(state.active_pellets),
        active_energizers=list(state.active_energizers),
    )

    direction = agent._predictor.predict(obs)
    assert direction.apply(obs.pacman_pos) in example_maze.pacman_graph[obs.pacman_pos]
