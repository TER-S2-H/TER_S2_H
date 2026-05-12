"""Blinky should chase the junction Pac-Man is heading toward when not frightened."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.iceguct_ghosts import IceGuctGhosts
from src.game.types import Direction, Observation


def _find_eastward_corridor(maze):
    for pos in maze.pacman_graph:
        if Direction.LEFT.apply(pos) not in maze.pacman_graph.get(pos, ()):
            continue
        if Direction.RIGHT.apply(pos) not in maze.pacman_graph.get(pos, ()):
            continue
        cur = pos
        while True:
            nxt = Direction.RIGHT.apply(cur)
            if nxt not in maze.pacman_graph.get(cur, ()):
                break
            cur = nxt
            if cur in maze.junctions and cur != pos:
                return pos, cur
    raise AssertionError("Could not find an eastbound corridor leading to a junction")


def test_blinky_chase(example_maze):
    maze = example_maze
    agent = IceGuctGhosts()
    agent.reset(maze, "ghosts")

    pacman_pos, target_junction = _find_eastward_corridor(maze)
    prev_pacman_pos = Direction.LEFT.apply(pacman_pos)
    ghost_pos = maze.ghost_spawns[0]

    agent._prev_obs = Observation(
        step=0,
        pacman_pos=prev_pacman_pos,
        ghost_positions=[ghost_pos] * 4,
        ghost_frightened=[0, 0, 0, 0],
        active_pellets=list(maze.initial_pellets),
        active_energizers=list(maze.initial_energizers),
    )
    obs = Observation(
        step=1,
        pacman_pos=pacman_pos,
        ghost_positions=[ghost_pos] * 4,
        ghost_frightened=[0, 0, 0, 0],
        active_pellets=list(maze.initial_pellets),
        active_energizers=list(maze.initial_energizers),
    )

    action = agent.act(obs)[0]
    before = maze.bfs_distances(ghost_pos, use_ghost_graph=True).get(target_junction, 10**9)
    after = maze.bfs_distances(action.apply(ghost_pos), use_ghost_graph=True).get(target_junction, 10**9)
    assert after < before
