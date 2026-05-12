"""The ICE gUCT trees should be rebuilt after an energizer-triggered global reverse."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.iceguct_ghosts import IceGuctGhosts
from src.game.simulator import SimState, step as sim_step
from src.game.types import Direction, Observation


class EnergizerChaser:
    name = "energizer_chaser"

    def reset(self, maze, role):
        self.maze = maze

    def act(self, obs):
        targets = set(obs.active_energizers) or set(obs.active_pellets)
        legal = [d for d in Direction.cardinals() if d.apply(obs.pacman_pos) in self.maze.pacman_graph.get(obs.pacman_pos, ())]
        if not legal:
            return Direction.STAY
        if targets:
            target = min(targets, key=lambda p: self.maze.bfs_distances(obs.pacman_pos).get(p, 10**9))
            path = self.maze.bfs_path(obs.pacman_pos, target)
            if path and len(path) > 1:
                dx, dy = path[1][0] - path[0][0], path[1][1] - path[0][1]
                return Direction.from_delta(dx, dy)
        return legal[0]


def test_tree_rebuild_on_global_reverse(maze):
    if not maze.initial_energizers:
        pytest.skip("This maze has no energizer, so the reverse trigger cannot be exercised.")

    pacman = EnergizerChaser()
    ghosts = IceGuctGhosts()
    pacman.reset(maze, "pacman")
    ghosts.reset(maze, "ghosts")

    energizer = next(iter(maze.initial_energizers))
    pacman_start = next(nb for nb in maze.pacman_graph.get(energizer, ()) if nb not in maze.ghost_house)
    state = SimState(
        pacman_pos=pacman_start,
        ghost_positions=list(maze.ghost_spawns),
        ghost_frightened=[0, 0, 0, 0],
        active_pellets=maze.initial_pellets,
        active_energizers=frozenset({energizer}),
    )
    event_seen = False
    rebuilds_before_event = None

    for _ in range(20):
        obs = Observation(
            step=state.step,
            pacman_pos=state.pacman_pos,
            ghost_positions=state.ghost_positions,
            ghost_frightened=state.ghost_frightened,
            active_pellets=list(state.active_pellets),
            active_energizers=list(state.active_energizers),
        )
        ghost_action = ghosts.act(obs)
        pac_action = pacman.act(obs)
        next_state = sim_step(state, maze, pac_action, ghost_action, max_steps=99999)

        if not event_seen and any(prev == 0 and curr > 0 for prev, curr in zip(state.ghost_frightened, next_state.ghost_frightened)):
            event_seen = True
            rebuilds_before_event = [brain.tree_rebuilds for brain in ghosts._brains[1:]]
        elif event_seen:
            assert rebuilds_before_event is not None
            assert ghosts._last_global_reverse is True
            after = [brain.tree_rebuilds for brain in ghosts._brains[1:]]
            assert all(a > b for a, b in zip(after, rebuilds_before_event))
            return

        state = next_state

    pytest.fail("Could not trigger an energizer event within 400 steps")
