from abc import ABC, abstractmethod
from enum import Enum, auto


class Direction(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    NONE = auto()


DIRECTION_VECTORS = {
    Direction.UP:    (0, -1),
    Direction.DOWN:  (0,  1),
    Direction.LEFT:  (-1, 0),
    Direction.RIGHT: (1,  0),
    Direction.NONE:  (0,  0),
}

OPPOSITE = {
    Direction.UP:    Direction.DOWN,
    Direction.DOWN:  Direction.UP,
    Direction.LEFT:  Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
    Direction.NONE:  Direction.NONE,
}


class Agent(ABC):
    @abstractmethod
    def get_action(self, game_state: dict) -> Direction:
        """Retourne la prochaine direction souhaitée par l'agent."""
