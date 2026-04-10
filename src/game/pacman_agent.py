import pygame
from .agent import Agent, Direction


class PacManAgent(Agent):
    """
    Agent Pac-Man contrôlé au clavier.

    Fonctionnement :
      1. handle_event(event) est appelé pour chaque événement pygame KEYDOWN.
         Il mémorise la direction souhaitée dans _buffered_direction.
      2. get_action() retourne _buffered_direction.
      3. Environment confirme si le mouvement a réussi via confirm_direction().
         Si oui, _buffered_direction est conservé (Pac-Man continue tout droit).
         Si non (mur), _buffered_direction reste en attente pour le prochain tick.

    Pour brancher un bot IA à la place du clavier :
      Créer une sous-classe et surcharger get_action(game_state) sans toucher
      à handle_event() ni à Game.
    """

    def __init__(self):
        self._buffered_direction: Direction = Direction.NONE
        self._current_direction: Direction = Direction.NONE

    # ------------------------------------------------------------------
    # Interface pygame
    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> None:
        """À appeler pour chaque pygame.KEYDOWN dans la boucle de jeu."""
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

    # ------------------------------------------------------------------
    # Interface Agent
    # ------------------------------------------------------------------

    def get_action(self, game_state: dict) -> Direction:
        return self._buffered_direction

    def confirm_direction(self, direction: Direction) -> None:
        """
        Appelé par Environment après chaque tick pour indiquer la direction
        effectivement utilisée. Permet de conserver le buffer si bloqué.
        """
        self._current_direction = direction
        # Si le mouvement a réussi, on continue dans cette direction par défaut
        if direction != Direction.NONE:
            self._buffered_direction = direction
