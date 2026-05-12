"""
Tick-based game simulator.

One tick = one cell of movement for every actor.
No randomness in step() — randomness lives in the agents/rollouts.
"""

from __future__ import annotations

from typing import FrozenSet, List, Optional, Tuple

from .types import Direction, GhostAction, PacmanAction, Pos
from .maze import Maze, FRIGHTENED_DURATION


class SimState:
    """
    Pure-data game state.  Designed to be copied cheaply for MCTS.

    active_pellets / active_energizers are frozensets so they are
    hashable and never accidentally mutated.
    """

    __slots__ = (
        "pacman_pos",
        "ghost_positions",
        "ghost_frightened",
        "active_pellets",
        "active_energizers",
        "step",
        "outcome",
        "energizer_penalty",  # True if energizer eaten while ghosts already frightened
    )

    def __init__(
        self,
        pacman_pos:        Pos,
        ghost_positions:   List[Pos],
        ghost_frightened:  List[int],
        active_pellets:    FrozenSet[Pos],
        active_energizers: FrozenSet[Pos],
        step:              int  = 0,
        outcome:           Optional[str] = None,
        energizer_penalty: bool = False,
    ):
        self.pacman_pos        = pacman_pos
        self.ghost_positions   = list(ghost_positions)
        self.ghost_frightened  = list(ghost_frightened)
        self.active_pellets    = frozenset(active_pellets)
        self.active_energizers = frozenset(active_energizers)
        self.step              = step
        self.outcome           = outcome
        self.energizer_penalty = energizer_penalty

    def copy(self) -> "SimState":
        s = SimState.__new__(SimState)
        s.pacman_pos        = self.pacman_pos
        s.ghost_positions   = list(self.ghost_positions)
        s.ghost_frightened  = list(self.ghost_frightened)
        s.active_pellets    = self.active_pellets    # frozenset — immutable
        s.active_energizers = self.active_energizers # frozenset — immutable
        s.step              = self.step
        s.outcome           = self.outcome
        s.energizer_penalty = self.energizer_penalty
        return s

    @property
    def is_terminal(self) -> bool:
        return self.outcome is not None

    def to_observation(self) -> dict:
        return {
            "step":              self.step,
            "pacman_pos":        self.pacman_pos,
            "ghost_positions":   list(self.ghost_positions),
            "ghost_frightened":  list(self.ghost_frightened),
            "active_pellets":    list(self.active_pellets),
            "active_energizers": list(self.active_energizers),
        }


def initial_state(maze: Maze) -> SimState:
    """Return the initial SimState for a maze."""
    return SimState(
        pacman_pos        = maze.pacman_spawn,
        ghost_positions   = list(maze.ghost_spawns),
        ghost_frightened  = [0, 0, 0, 0],
        active_pellets    = maze.initial_pellets,
        active_energizers = maze.initial_energizers,
    )


def _move(pos: Pos, direction: Direction, graph: dict) -> Pos:
    """Move one step; return new position (stay if blocked)."""
    if direction == Direction.STAY:
        return pos
    nb = direction.apply(pos)
    if nb in graph.get(pos, ()):
        return nb
    return pos   # blocked — stay


def step(
    state: SimState,
    maze: Maze,
    pacman_action: PacmanAction,
    ghost_actions: GhostAction,
    frightened_duration: int = FRIGHTENED_DURATION,
    max_steps: int = 5000,
) -> SimState:
    """
    Advance game state by one tick.  Returns a NEW SimState.

    Movement is simultaneous; collisions are checked after.
    """
    if state.is_terminal:
        return state.copy()

    # ---- Move Pac-Man ------------------------------------------------
    pacman_new = _move(state.pacman_pos, pacman_action, maze.pacman_graph)

    # ---- Move ghosts -------------------------------------------------
    ghost_new: List[Pos] = []
    for i, (gpos, gdir) in enumerate(zip(state.ghost_positions, ghost_actions)):
        ghost_new.append(_move(gpos, gdir, maze.ghost_graph))

    # ---- Decrement frightened timers ---------------------------------
    new_frightened = [max(0, f - 1) for f in state.ghost_frightened]

    # ---- Consume pellets / energizers --------------------------------
    new_pellets    = state.active_pellets
    new_energizers = state.active_energizers
    energizer_penalty = False

    if pacman_new in new_pellets:
        new_pellets = new_pellets - {pacman_new}

    if pacman_new in new_energizers:
        new_energizers = new_energizers - {pacman_new}
        if any(f > 0 for f in state.ghost_frightened):
            # Energizer eaten while ≥1 ghost already frightened → penalty (§5.4)
            energizer_penalty = True
        else:
            new_frightened = [frightened_duration] * 4

    # ---- Collision detection -----------------------------------------
    new_outcome: Optional[str] = None
    new_ghost_frightened = new_frightened[:]
    new_ghost_positions  = ghost_new[:]

    for i in range(4):
        gpos_new = ghost_new[i]
        gpos_old = state.ghost_positions[i]

        # Overlap collision: same cell after move
        overlap = (gpos_new == pacman_new)
        # Crossing collision: Pac-Man and ghost swapped cells
        crossing = (gpos_old == pacman_new and gpos_new == state.pacman_pos)

        if overlap or crossing:
            if new_frightened[i] > 0 or state.ghost_frightened[i] > 0:
                # Ghost eaten by Pac-Man → respawn
                new_ghost_positions[i]  = maze.ghost_spawns[i]
                new_ghost_frightened[i] = 0
            else:
                new_outcome = "eaten"
                break  # Pac-Man dies — stop processing

    new_step = state.step + 1
    if new_outcome is None:
        if not new_pellets and not new_energizers:
            new_outcome = "cleared"
        elif new_step >= max_steps:
            new_outcome = "timeout"

    return SimState(
        pacman_pos        = pacman_new,
        ghost_positions   = new_ghost_positions,
        ghost_frightened  = new_ghost_frightened,
        active_pellets    = new_pellets,
        active_energizers = new_energizers,
        step              = new_step,
        outcome           = new_outcome,
        energizer_penalty = energizer_penalty,
    )
