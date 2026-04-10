from abc import ABC, abstractmethod
from enum import Enum, auto


class Direction(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    NONE = auto()


# Vecteurs de déplacement associés à chaque direction
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
    """
    Classe abstraite représentant tout agent du jeu (Pac-Man ou fantôme).

    game_state reçu par get_action() est un dict contenant :
      - "walls"         : set of (x, y)
      - "ghost_walls"   : set of (x, y)
      - "width", "height"
      - "pacman"        : {"x", "y", "dir"}
      - "ghosts"        : list of {"x", "y", "frightened", "id"}
      - "pellets"       : dict (x, y) -> Pellet
      - "score", "lives", "level"
      - "frightened_timer"
    """

    @abstractmethod
    def get_action(self, game_state: dict) -> Direction:
        """Retourne la prochaine direction souhaitée par l'agent."""
