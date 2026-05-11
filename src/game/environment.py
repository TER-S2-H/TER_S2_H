import random
import requests
from typing import Optional

from .agent import Direction, DIRECTION_VECTORS, OPPOSITE
from .pellet import Pellet
from .ghost import Ghost

# URL de l'API déployée sur Render
CLOUD_API_URL = "https://pacman-s2-h.onrender.com"
LOCAL_API_URL = "http://localhost:5000"

# Positions classiques des power pellets (coins du labyrinthe 28×31)
POWER_PELLET_CANDIDATES = {(1, 3), (26, 3), (1, 23), (26, 23)}

# Positions de départ des fantômes
GHOST_START_POSITIONS = [
    (13, 11),  # Blinky – juste au-dessus de la maison
    (13, 14),  # Pinky  – intérieur
    (11, 14),  # Inky   – intérieur gauche
    (15, 14),  # Clyde  – intérieur droite
]

# Durée du mode frightened (en ticks, à 10 ticks/s → 7 s)
FRIGHTENED_DURATION = 70

# Points pour manger un fantôme enchaîné : 200, 400, 800, 1600
GHOST_EAT_BASE = 200


class Environment:
    """
    État complet du plateau de jeu.

    Responsabilités
    ---------------
    - Charger le labyrinthe depuis l'API Flask (load_maze)
    - Placer les pellets sur les nœuds PATH
    - Mettre à jour l'état à chaque tick (update)
    - Détecter les collisions et renvoyer la liste des événements survenus
    - Exposer get_state() pour les agents et get_grid_snapshot() pour le Recorder

    Agents
    ------
    self.ghosts est une liste d'objets Ghost (ou sous-classes).
    Game injecte ces objets après load_maze() :
        env.ghosts = [Ghost(i, *pos) for i, pos in enumerate(GHOST_START_POSITIONS)]
    Ainsi, remplacer un Ghost par une sous-classe IA ne nécessite aucune
    modification de cette classe.
    """

    def __init__(self, seed: Optional[int] = None, base_url: str = CLOUD_API_URL):
        self.base_url  = base_url
        self.seed      = seed

        # Grille
        self.width:  int = 28
        self.height: int = 31
        self.walls:       set = set()   # set of (x, y)
        self.ghost_walls: set = set()   # set of (x, y)
        self._path_nodes: set = set()   # PATH nodes → reçoivent des pellets
        self._blank_nodes: set = set()  # PATHBLANK → pas de pellets (ghost house…)

        # Entités
        self.pellets: dict = {}         # (x, y) -> Pellet
        self.ghosts:  list = []         # injecté par Game

        # Pac-Man
        self.pacman_x:   int = 13
        self.pacman_y:   int = 23
        self.pacman_dir: Direction = Direction.NONE

        # Score / état
        self.score:    int = 0
        self.lives:    int = 3
        self.level:    int = 1
        self.game_over:     bool = False
        self.level_complete: bool = False

        # Mode frightened
        self.frightened_timer:  int = 0
        self.ghost_eat_streak:  int = 0   # réinitialisé à chaque power pellet

        # Données brutes de l'API (conservées pour le Recorder)
        self.maze_data: Optional[dict] = None

    # ------------------------------------------------------------------
    # Chargement du labyrinthe
    # ------------------------------------------------------------------

    def load_maze(self) -> None:
        """Appelle l'API /maze, construit la grille et place les pellets."""
        params = {}
        if self.seed is not None:
            params["seed"] = self.seed

        resp = requests.get(f"{self.base_url}/maze", params=params, timeout=60)
        resp.raise_for_status()
        self.maze_data = resp.json()
        self.seed = self.maze_data.get("seed", self.seed)

        meta = self.maze_data.get("metadata", {})
        self.width  = meta.get("width",  28)
        self.height = meta.get("height", 31)

        self._build_grid()
        self._place_pellets()

    def _build_grid(self) -> None:
        self.walls       = {(w["x"], w["y"]) for w in self.maze_data.get("walls", [])}
        self.ghost_walls = {(g["x"], g["y"]) for g in self.maze_data.get("ghost_walls", [])}
        self._path_nodes  = set()
        self._blank_nodes = set()
        for node in self.maze_data.get("nodes", []):
            pos = (node["x"], node["y"])
            if node["type"] == "PATH":
                self._path_nodes.add(pos)
            elif node["type"] == "PATHBLANK":
                self._blank_nodes.add(pos)

    def _place_pellets(self) -> None:
        """Place des pellets sur tous les nœuds PATH (power pellets aux coins)."""
        self.pellets = {}
        for pos in self._path_nodes:
            kind = "power" if pos in POWER_PELLET_CANDIDATES else "normal"
            self.pellets[pos] = Pellet(pos[0], pos[1], kind)
        # Retirer le pellet à la position de départ de Pac-Man
        self.pellets.pop((self.pacman_x, self.pacman_y), None)

    # ------------------------------------------------------------------
    # Déplacement
    # ------------------------------------------------------------------

    def _is_wall(self, x: int, y: int, for_ghost: bool = False) -> bool:
        """Retourne True si la case (x, y) bloque le mouvement."""
        if y < 0 or y >= self.height:
            return True
        pos = (x % self.width, y)    # gestion du tunnel horizontal
        if pos in self.walls:
            return True
        # La porte de la maison des fantômes bloque Pac-Man, pas les fantômes
        if not for_ghost and pos in self.ghost_walls:
            return True
        return False

    def _try_move(self, x: int, y: int, direction: Direction,
                  for_ghost: bool = False) -> tuple:
        """
        Tente un déplacement d'une case dans direction.
        Retourne (new_x, new_y, direction_used) ;
        direction_used vaut Direction.NONE si bloqué.
        """
        dx, dy = DIRECTION_VECTORS[direction]
        nx = (x + dx) % self.width
        ny = y + dy
        if not self._is_wall(nx, ny, for_ghost=for_ghost):
            return nx, ny, direction
        return x, y, Direction.NONE

    # ------------------------------------------------------------------
    # Mise à jour principale
    # ------------------------------------------------------------------

    def update(self, actions: dict) -> list:
        """
        Fait avancer l'état du jeu d'un tick.

        Paramètres
        ----------
        actions : dict
            {"pacman": Direction, "ghost_0": Direction, "ghost_1": Direction, …}

        Retour
        ------
        events : list[str]
            Liste des événements survenus ce tick (pellet_normal, pellet_power,
            ghost_eaten_200, pacman_dead, game_over, level_complete…)
        """
        events = []

        # ── Déplacement Pac-Man ────────────────────────────────────────
        wanted = actions.get("pacman", Direction.NONE)
        nx, ny, moved = self._try_move(self.pacman_x, self.pacman_y, wanted)
        if moved == Direction.NONE and self.pacman_dir != Direction.NONE:
            # Essayer de continuer dans la direction actuelle
            nx, ny, moved = self._try_move(self.pacman_x, self.pacman_y, self.pacman_dir)
        self.pacman_x, self.pacman_y = nx, ny
        if moved != Direction.NONE:
            self.pacman_dir = moved

        # ── Collecte de pellets ────────────────────────────────────────
        pos = (self.pacman_x, self.pacman_y)
        if pos in self.pellets:
            pellet = self.pellets.pop(pos)
            self.score += pellet.score
            events.append(f"pellet_{pellet.type}")
            if pellet.type == "power":
                self.frightened_timer = FRIGHTENED_DURATION
                self.ghost_eat_streak = 0
                for g in self.ghosts:
                    g.frightened = True
                events.append("power_pellet")

        # ── Déplacement des fantômes ───────────────────────────────────
        for i, ghost in enumerate(self.ghosts):
            preferred = actions.get(f"ghost_{i}", Direction.NONE)
            gx, gy, gmoved = self._try_move(ghost.x, ghost.y, preferred, for_ghost=True)

            if gmoved == Direction.NONE:
                # Direction préférée bloquée → essayer d'autres (sans demi-tour)
                opposite = OPPOSITE[ghost._last_direction]
                candidates = [d for d in (Direction.UP, Direction.DOWN,
                                          Direction.LEFT, Direction.RIGHT)
                              if d != opposite and d != preferred]
                random.shuffle(candidates)
                for d in candidates:
                    gx, gy, gmoved = self._try_move(ghost.x, ghost.y, d, for_ghost=True)
                    if gmoved != Direction.NONE:
                        break

            if gmoved == Direction.NONE:
                # Dernier recours : demi-tour
                rev = OPPOSITE[ghost._last_direction]
                gx, gy, gmoved = self._try_move(ghost.x, ghost.y, rev, for_ghost=True)

            ghost.x, ghost.y = gx, gy
            if gmoved != Direction.NONE:
                ghost._last_direction = gmoved

        # ── Décompte frightened ────────────────────────────────────────
        if self.frightened_timer > 0:
            self.frightened_timer -= 1
            if self.frightened_timer == 0:
                for g in self.ghosts:
                    g.frightened = False
                self.ghost_eat_streak = 0

        # ── Collisions Pac-Man / fantômes ─────────────────────────────
        for ghost in self.ghosts:
            if ghost.x == self.pacman_x and ghost.y == self.pacman_y:
                if ghost.frightened:
                    self.ghost_eat_streak += 1
                    pts = GHOST_EAT_BASE * (2 ** (self.ghost_eat_streak - 1))
                    self.score += pts
                    ghost.reset()
                    events.append(f"ghost_eaten_{pts}")
                else:
                    events.append("pacman_dead")
                    self.lives -= 1
                    if self.lives <= 0:
                        self.game_over = True
                        events.append("game_over")
                    else:
                        self._reset_positions()
                    break   # une mort suffit

        # ── Niveau terminé ─────────────────────────────────────────────
        if not self.pellets and not self.level_complete:
            self.level_complete = True
            events.append("level_complete")

        return events

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_positions(self) -> None:
        """Replace Pac-Man et les fantômes après une mort."""
        self.pacman_x   = 13
        self.pacman_y   = 23
        self.pacman_dir = Direction.NONE
        for g in self.ghosts:
            g.reset()
        self.frightened_timer = 0
        self.ghost_eat_streak = 0

    def next_level(self) -> None:
        """Passe au niveau suivant : repose les pellets et replace les agents."""
        self.level += 1
        self.level_complete = False
        self._place_pellets()
        self._reset_positions()

    # ------------------------------------------------------------------
    # Exposition de l'état
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Retourne un instantané de l'état lisible par les agents."""
        return {
            "walls":           self.walls,
            "ghost_walls":     self.ghost_walls,
            "width":           self.width,
            "height":          self.height,
            "pacman":          {"x": self.pacman_x, "y": self.pacman_y,
                                "dir": self.pacman_dir},
            "ghosts":          [{"x": g.x, "y": g.y, "frightened": g.frightened,
                                 "id": g.ghost_id, "name": g.name}
                                for g in self.ghosts],
            "pellets":         self.pellets,
            "score":           self.score,
            "lives":           self.lives,
            "level":           self.level,
            "frightened_timer": self.frightened_timer,
        }

    def get_grid_snapshot(self) -> list:
        """
        Retourne le plateau sous forme de liste 2D de caractères.
        Utilisé par Recorder pour la sauvegarde.
          'W' = mur, ' ' = couloir vide, 'o' = pellet normal,
          'O' = power pellet, 'P' = Pac-Man, 'G' = fantôme
        """
        grid = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if (x, y) in self.walls:
                    row.append('W')
                elif (x, y) in self.ghost_walls:
                    row.append('D')   # Door
                else:
                    row.append(' ')
            grid.append(row)

        for (x, y), p in self.pellets.items():
            grid[y][x] = 'o' if p.type == 'normal' else 'O'

        grid[self.pacman_y][self.pacman_x] = 'P'
        for g in self.ghosts:
            if 0 <= g.y < self.height and 0 <= g.x < self.width:
                grid[g.y][g.x] = 'G'

        return grid
