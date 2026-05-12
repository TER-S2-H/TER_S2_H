"""ICE gUCT should only emit legal ghost moves while navigating out of the house."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.iceguct_ghosts import IceGuctGhosts
from src.game.simulator import initial_state, step as sim_step
from src.game.types import Direction, Observation


def test_iceguct_ghost_actions_are_legal(maze):
    agent = IceGuctGhosts(rng=np.random.default_rng(0))
    agent.reset(maze, "ghosts")

    state = initial_state(maze)
    for _ in range(8):
        obs = Observation(
            step=state.step,
            pacman_pos=state.pacman_pos,
            ghost_positions=state.ghost_positions,
            ghost_frightened=state.ghost_frightened,
            active_pellets=list(state.active_pellets),
            active_energizers=list(state.active_energizers),
        )
        actions = agent.act(obs)

        for pos, direction in zip(state.ghost_positions, actions):
            assert direction == Direction.STAY or direction.apply(pos) in maze.ghost_graph.get(pos, ())

        state = sim_step(state, maze, Direction.STAY, actions)
