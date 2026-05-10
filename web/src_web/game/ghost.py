import random
from .agent import Agent, Direction, OPPOSITE

GHOST_COLORS = [
    (255,   0,   0),
    (255, 182, 255),
    (  0, 255, 255),
    (255, 165,   0),
]

GHOST_NAMES = ["Blinky", "Pinky", "Inky", "Clyde"]

FRIGHTENED_COLOR = (0, 0, 200)


class Ghost(Agent):
    def __init__(self, ghost_id: int, start_x: int, start_y: int):
        self.ghost_id  = ghost_id
        self.name      = GHOST_NAMES[ghost_id % len(GHOST_NAMES)]
        self.color     = GHOST_COLORS[ghost_id % len(GHOST_COLORS)]
        self.start_x   = start_x
        self.start_y   = start_y
        self.x         = start_x
        self.y         = start_y
        self.frightened        = False
        self._last_direction   = Direction.NONE

    def get_action(self, game_state: dict) -> Direction:
        candidates = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        opposite = OPPOSITE[self._last_direction]
        preferred = [d for d in candidates if d != opposite]
        return random.choice(preferred if preferred else candidates)

    def reset(self) -> None:
        self.x = self.start_x
        self.y = self.start_y
        self.frightened      = False
        self._last_direction = Direction.NONE

    def __repr__(self):
        return f"Ghost({self.name}, x={self.x}, y={self.y}, frightened={self.frightened})"


_GHOST_REGISTRY: dict = {
    "random": Ghost,
}


def get_ghost_agent(name: str, ghost_id: int,
                    start_x: int, start_y: int) -> Ghost:
    cls = _GHOST_REGISTRY.get(name, Ghost)
    return cls(ghost_id, start_x, start_y)
