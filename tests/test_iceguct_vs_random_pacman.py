"""ICE gUCT should comfortably beat a random Pac-Man on the example maze."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents import RandomPacmanAgent
from src.agents.iceguct_ghosts import IceGuctGhosts
from src.benchmark.run_game import run_game


def test_iceguct_vs_random_pacman(example_maze):
    wins = 0
    for seed in range(10):
        record = run_game(
            example_maze,
            RandomPacmanAgent(),
            IceGuctGhosts(),
            seed=seed,
            max_steps=250,
            record=False,
        )
        wins += int(record["outcome"] == "eaten")
    assert wins >= 8
