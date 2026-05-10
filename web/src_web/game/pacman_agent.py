import pygame
from .agent import Agent, Direction


class ReplayPacManAgent(Agent):
    def __init__(self, actions: list):
        self._actions = actions
        self._index   = 0

    def handle_event(self, event) -> None:
        pass

    def get_action(self, game_state: dict) -> Direction:
        if self._index >= len(self._actions):
            return Direction.NONE
        name = self._actions[self._index]
        self._index += 1
        try:
            return Direction[name]
        except KeyError:
            return Direction.NONE

    def confirm_direction(self, direction: Direction) -> None:
        pass


class PacManAgent(Agent):
    def __init__(self):
        self._buffered_direction: Direction = Direction.NONE
        self._current_direction: Direction = Direction.NONE

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        mapping = {
            pygame.K_UP:    Direction.UP,
            pygame.K_w:     Direction.UP,
            pygame.K_DOWN:  Direction.DOWN,
            pygame.K_s:     Direction.DOWN,
            pygame.K_LEFT:  Direction.LEFT,
            pygame.K_a:     Direction.LEFT,
            pygame.K_RIGHT: Direction.RIGHT,
            pygame.K_d:     Direction.RIGHT,
        }
        if event.key in mapping:
            self._buffered_direction = mapping[event.key]

    def get_action(self, game_state: dict) -> Direction:
        return self._buffered_direction

    def confirm_direction(self, direction: Direction) -> None:
        self._current_direction = direction
        if direction != Direction.NONE:
            self._buffered_direction = direction
