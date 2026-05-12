"""Agent protocol — the only interface JEPA agents need to implement."""

from __future__ import annotations

from typing import Callable, Literal, Protocol, runtime_checkable

from ..game.types import GhostAction, Observation, PacmanAction


@runtime_checkable
class Agent(Protocol):
    name: str

    def reset(self, maze, role: Literal["pacman", "ghosts"]) -> None:
        """Called once before each game. maze is a Maze instance."""
        ...

    def act(self, observation: Observation) -> "PacmanAction | GhostAction":
        """
        Called every tick.
        For role='pacman' return a Direction (PacmanAction).
        For role='ghosts' return a 4-tuple of Directions (GhostAction).
        """
        ...


AgentFactory = Callable[[], Agent]
