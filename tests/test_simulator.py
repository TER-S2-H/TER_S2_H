"""Basic simulator sanity checks."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.game.simulator import SimState, initial_state, step as sim_step
from src.game.types import Direction


def test_initial_state(example_maze):
    m = example_maze
    s = initial_state(m)
    assert s.pacman_pos == m.pacman_spawn
    assert len(s.ghost_positions) == 4
    assert s.active_pellets    == m.initial_pellets
    assert s.active_energizers == m.initial_energizers
    assert not s.is_terminal


def test_copy_independence(example_maze):
    m  = example_maze
    s1 = initial_state(m)
    s2 = s1.copy()
    s2.ghost_positions[0] = (0, 0)
    assert s1.ghost_positions[0] != (0, 0)


def test_pacman_cannot_enter_ghost_house(example_maze):
    m = example_maze
    s = initial_state(m)

    # Move Pac-Man: ghost_house cells must never appear in pacman_graph
    pac_nodes = set(m.pacman_graph.keys())
    assert all(p not in m.ghost_house for p in pac_nodes)


def test_step_eats_pellet(example_maze):
    m = example_maze
    s = initial_state(m)

    # Find a pellet adjacent to spawn
    spawn = m.pacman_spawn
    pellet_nbrs = [
        nb for nb in m.pacman_graph.get(spawn, ())
        if nb in m.initial_pellets
    ]
    if not pellet_nbrs:
        pytest.skip("No pellet adjacent to spawn in this maze")

    target = pellet_nbrs[0]
    dx, dy = target[0] - spawn[0], target[1] - spawn[1]
    d = Direction.from_delta(dx, dy)

    g_action = (Direction.STAY,) * 4
    s2 = sim_step(s, m, d, g_action)

    assert s2.pacman_pos == target
    assert target not in s2.active_pellets


def test_outcome_cleared(example_maze):
    m = example_maze
    # Create state with just one pellet remaining adjacent to spawn
    spawn = m.pacman_spawn
    nbrs  = [nb for nb in m.pacman_graph.get(spawn, ()) if nb in m.initial_pellets]
    if not nbrs:
        pytest.skip("No pellet adjacent to spawn")

    target = nbrs[0]
    s = SimState(
        pacman_pos        = spawn,
        ghost_positions   = list(m.ghost_spawns),
        ghost_frightened  = [0, 0, 0, 0],
        active_pellets    = frozenset({target}),
        active_energizers = frozenset(),
    )
    dx, dy = target[0] - spawn[0], target[1] - spawn[1]
    d = Direction.from_delta(dx, dy)
    s2 = sim_step(s, m, d, (Direction.STAY,) * 4)
    assert s2.outcome == "cleared"
