"""KNN prediction should return a legal direction quickly on a synthetic dataset."""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.iceguct_ghosts import KNNPredictor
from src.game.simulator import initial_state
from src.game.types import Observation


def test_knn_prediction_shape(example_maze):
    maze = example_maze
    x = np.stack([
        np.linspace(0.0, 1.0, 10, dtype=np.float32) + i * 0.01
        for i in range(20)
    ])
    y = np.roll(x, shift=1, axis=1)
    predictor = KNNPredictor(maze, x, y, rng=np.random.default_rng(1))

    state = initial_state(maze)
    obs = Observation(
        step=state.step,
        pacman_pos=state.pacman_pos,
        ghost_positions=state.ghost_positions,
        ghost_frightened=state.ghost_frightened,
        active_pellets=list(state.active_pellets),
        active_energizers=list(state.active_energizers),
    )

    start = time.perf_counter()
    direction = predictor.predict(obs)
    elapsed = time.perf_counter() - start

    assert direction.apply(obs.pacman_pos) in maze.pacman_graph[obs.pacman_pos]
    assert elapsed < 0.05
