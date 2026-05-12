"""Random agents — choose a legal move uniformly at random each tick."""

from __future__ import annotations

import random
from typing import Literal

from ..game.maze import Maze
from ..game.types import Direction, GhostAction, Observation, PacmanAction


class RandomPacmanAgent:
    name = "random_pacman"

    def reset(self, maze: Maze, role: Literal["pacman", "ghosts"]) -> None:
        self._maze = maze

    def act(self, obs: Observation) -> PacmanAction:
        pos   = obs.pacman_pos
        legal = [
            d for d in Direction.cardinals()
            if d.apply(pos) in self._maze.pacman_graph.get(pos, ())
        ]
        return random.choice(legal) if legal else Direction.STAY


class RandomGhostsAgent:
    name = "random_ghosts"

    def reset(self, maze: Maze, role: Literal["pacman", "ghosts"]) -> None:
        self._maze = maze
        self._prev_dirs = [Direction.STAY] * 4

    def act(self, obs: Observation) -> GhostAction:
        dirs = []
        for i, gpos in enumerate(obs.ghost_positions):
            reverse = self._prev_dirs[i].reverse()
            legal = [
                d for d in Direction.cardinals()
                if d.apply(gpos) in self._maze.ghost_graph.get(gpos, ())
                and d != reverse
            ]
            if not legal:
                legal = [
                    d for d in Direction.cardinals()
                    if d.apply(gpos) in self._maze.ghost_graph.get(gpos, ())
                ]
            chosen = random.choice(legal) if legal else Direction.STAY
            dirs.append(chosen)
            self._prev_dirs[i] = chosen
        return tuple(dirs)  # type: ignore
