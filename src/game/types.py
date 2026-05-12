"""Core types shared across the Pac-Man MCTS framework."""

from __future__ import annotations
from enum import Enum
from typing import Tuple, List, Optional
from dataclasses import dataclass, field


Pos = Tuple[int, int]


class Direction(Enum):
    UP    = (0, -1)
    DOWN  = (0,  1)
    LEFT  = (-1, 0)
    RIGHT = (1,  0)
    STAY  = (0,  0)

    @property
    def dx(self) -> int:
        return self.value[0]

    @property
    def dy(self) -> int:
        return self.value[1]

    def apply(self, pos: Pos) -> Pos:
        return (pos[0] + self.value[0], pos[1] + self.value[1])

    def reverse(self) -> "Direction":
        dx, dy = self.value
        return Direction((-dx, -dy))

    @staticmethod
    def from_delta(dx: int, dy: int) -> "Direction":
        for d in Direction:
            if d.value == (dx, dy):
                return d
        raise ValueError(f"No direction for delta ({dx}, {dy})")

    @staticmethod
    def cardinals() -> List["Direction"]:
        return [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


PacmanAction = Direction
GhostAction  = Tuple[Direction, Direction, Direction, Direction]


@dataclass
class Observation:
    """Matches one frame from the session schema (§2)."""
    step:              int
    pacman_pos:        Pos
    ghost_positions:   List[Pos]
    ghost_frightened:  List[int]          # per-ghost remaining frightened ticks
    active_pellets:    List[Pos]
    active_energizers: List[Pos]

    def to_dict(self) -> dict:
        return {
            "step":              self.step,
            "pacman_pos":        self.pacman_pos,
            "ghost_positions":   list(self.ghost_positions),
            "ghost_frightened":  list(self.ghost_frightened),
            "active_pellets":    list(self.active_pellets),
            "active_energizers": list(self.active_energizers),
        }

    @staticmethod
    def from_dict(d: dict) -> "Observation":
        return Observation(
            step              = d["step"],
            pacman_pos        = tuple(d["pacman_pos"]),
            ghost_positions   = [tuple(p) for p in d["ghost_positions"]],
            ghost_frightened  = list(d["ghost_frightened"]),
            active_pellets    = [tuple(p) for p in d["active_pellets"]],
            active_energizers = [tuple(p) for p in d["active_energizers"]],
        )
