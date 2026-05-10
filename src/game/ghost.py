import random
from .agent import Agent, Direction, OPPOSITE

# Couleurs associées à chaque fantôme (index 0-3)
GHOST_COLORS = [
    (255,   0,   0),   # Blinky  – rouge
    (255, 182, 255),   # Pinky   – rose
    (  0, 255, 255),   # Inky    – cyan
    (255, 165,   0),   # Clyde   – orange
]

GHOST_NAMES = ["Blinky", "Pinky", "Inky", "Clyde"]

# Couleur en mode frightened
FRIGHTENED_COLOR = (0, 0, 200)


class Ghost(Agent):
    """
    Fantôme avec déplacement aléatoire.

    ── Comment brancher une IA ─────────────────────────────────────────
    Créer une sous-classe et surcharger get_action(game_state).
    game_state contient toutes les infos nécessaires au pathfinding :
      walls, ghost_walls, width, height, pacman, ghosts, pellets…

    Exemple minimal :
        class MyAIGhost(Ghost):
            def get_action(self, game_state: dict) -> Direction:
                # Implémenter BFS/A* ici
                return best_direction

    Ensuite dans main.py / game.py :
        ghosts[0] = MyAIGhost(0, *GHOST_START_POSITIONS[0])
    ────────────────────────────────────────────────────────────────────
    """

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

    # ------------------------------------------------------------------
    # Interface Agent — à surcharger dans les sous-classes IA
    # ------------------------------------------------------------------

    def get_action(self, game_state: dict) -> Direction:
        """
        Retourne une direction aléatoire.
        Évite le demi-tour sauf si c'est la seule option.

        Surcharger cette méthode pour implémenter une IA.
        Le paramètre game_state donne accès à tout l'état du jeu.
        """
        candidates = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        opposite = OPPOSITE[self._last_direction]
        preferred = [d for d in candidates if d != opposite]
        return random.choice(preferred if preferred else candidates)

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Replace le fantôme à sa position de départ (après qu'il est mangé)."""
        self.x = self.start_x
        self.y = self.start_y
        self.frightened      = False
        self._last_direction = Direction.NONE

    def __repr__(self):
        return f"Ghost({self.name}, x={self.x}, y={self.y}, frightened={self.frightened})"


# ---------------------------------------------------------------------------
# Factory — permet aux binômes d'enregistrer leur IA sans toucher à game.py
# ---------------------------------------------------------------------------

# Registre des IAs disponibles : nom → classe Ghost (ou sous-classe)
_GHOST_REGISTRY: dict = {
    "random": Ghost,
    # Exemple pour brancher une IA :
    #   from .my_ai_ghost import BlinkyAI
    #   _GHOST_REGISTRY["blinky_ai"] = BlinkyAI
}


def get_ghost_agent(name: str, ghost_id: int,
                    start_x: int, start_y: int) -> Ghost:
    """
    Instancie un fantôme selon le nom de l'IA demandée.

    Usage dans main.py : --ai random  (seule valeur actuellement)

    Pour brancher une IA :
      1. Créer MyAIGhost(Ghost) dans src/game/my_ai_ghost.py
      2. Importer et ajouter dans _GHOST_REGISTRY ci-dessus
      3. Lancer : python main.py --ai mon_ia
    """
    cls = _GHOST_REGISTRY.get(name, Ghost)
    return cls(ghost_id, start_x, start_y)
