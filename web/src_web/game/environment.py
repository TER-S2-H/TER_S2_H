import random
from typing import Optional

from .agent import Direction, DIRECTION_VECTORS, OPPOSITE
from .pellet import Pellet
from .ghost import Ghost

CLOUD_API_URL = "https://pacman-s2-h.onrender.com"
LOCAL_API_URL = "http://localhost:5000"

POWER_PELLET_CANDIDATES = {(1, 3), (26, 3), (1, 23), (26, 23)}

GHOST_START_POSITIONS = [
    (13, 11),
    (13, 14),
    (11, 14),
    (15, 14),
]

FRIGHTENED_DURATION = 70
GHOST_EAT_BASE = 200


class Environment:
    """
    Version web : load_maze() est async (remplace requests par pyodide.http.pyfetch).
    Toute la logique de jeu est identique à src/game/environment.py.
    """

    def __init__(self, seed: Optional[int] = None, base_url: str = CLOUD_API_URL):
        self.base_url  = base_url
        self.seed      = seed

        self.width:  int = 28
        self.height: int = 31
        self.walls:       set = set()
        self.ghost_walls: set = set()
        self._path_nodes: set = set()
        self._blank_nodes: set = set()

        self.pellets: dict = {}
        self.ghosts:  list = []

        self.pacman_x:   int = 13
        self.pacman_y:   int = 23
        self.pacman_dir: Direction = Direction.NONE

        self.score:    int = 0
        self.lives:    int = 3
        self.level:    int = 1
        self.game_over:     bool = False
        self.level_complete: bool = False

        self.frightened_timer:  int = 0
        self.ghost_eat_streak:  int = 0

        self.maze_data: Optional[dict] = None

    # ------------------------------------------------------------------
    # Chargement asynchrone du labyrinthe
    # ------------------------------------------------------------------

    async def load_maze(self) -> None:
        """
        Appelle l'API /maze de façon asynchrone.
        - Navigateur (Pygbag) : utilise pyodide.http.pyfetch
        - Desktop (tests hors navigateur) : utilise aiohttp comme fallback
        """
        url = f"{self.base_url}/maze"
        if self.seed is not None:
            url += f"?seed={self.seed}"

        try:
            # Environnement Pyodide / Pygbag
            from pyodide.http import pyfetch  # type: ignore
            response = await pyfetch(url)
            self.maze_data = await response.json()
        except ImportError:
            # Fallback desktop (tests asyncio locaux avec aiohttp)
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    resp.raise_for_status()
                    self.maze_data = await resp.json()

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
        self.pellets = {}
        for pos in self._path_nodes:
            kind = "power" if pos in POWER_PELLET_CANDIDATES else "normal"
            self.pellets[pos] = Pellet(pos[0], pos[1], kind)
        self.pellets.pop((self.pacman_x, self.pacman_y), None)

    # ------------------------------------------------------------------
    # Déplacement
    # ------------------------------------------------------------------

    def _is_wall(self, x: int, y: int, for_ghost: bool = False) -> bool:
        if y < 0 or y >= self.height:
            return True
        pos = (x % self.width, y)
        if pos in self.walls:
            return True
        if not for_ghost and pos in self.ghost_walls:
            return True
        return False

    def _try_move(self, x: int, y: int, direction: Direction,
                  for_ghost: bool = False) -> tuple:
        dx, dy = DIRECTION_VECTORS[direction]
        nx = (x + dx) % self.width
        ny = y + dy
        if not self._is_wall(nx, ny, for_ghost=for_ghost):
            return nx, ny, direction
        return x, y, Direction.NONE

    # ------------------------------------------------------------------
    # Mise à jour principale (identique à l'original)
    # ------------------------------------------------------------------

    def update(self, actions: dict) -> list:
        events = []

        wanted = actions.get("pacman", Direction.NONE)
        nx, ny, moved = self._try_move(self.pacman_x, self.pacman_y, wanted)
        if moved == Direction.NONE and self.pacman_dir != Direction.NONE:
            nx, ny, moved = self._try_move(self.pacman_x, self.pacman_y, self.pacman_dir)
        self.pacman_x, self.pacman_y = nx, ny
        if moved != Direction.NONE:
            self.pacman_dir = moved

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

        for i, ghost in enumerate(self.ghosts):
            preferred = actions.get(f"ghost_{i}", Direction.NONE)
            gx, gy, gmoved = self._try_move(ghost.x, ghost.y, preferred, for_ghost=True)

            if gmoved == Direction.NONE:
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
                rev = OPPOSITE[ghost._last_direction]
                gx, gy, gmoved = self._try_move(ghost.x, ghost.y, rev, for_ghost=True)

            ghost.x, ghost.y = gx, gy
            if gmoved != Direction.NONE:
                ghost._last_direction = gmoved

        if self.frightened_timer > 0:
            self.frightened_timer -= 1
            if self.frightened_timer == 0:
                for g in self.ghosts:
                    g.frightened = False
                self.ghost_eat_streak = 0

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
                    break

        if not self.pellets and not self.level_complete:
            self.level_complete = True
            events.append("level_complete")

        return events

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_positions(self) -> None:
        self.pacman_x   = 13
        self.pacman_y   = 23
        self.pacman_dir = Direction.NONE
        for g in self.ghosts:
            g.reset()
        self.frightened_timer = 0
        self.ghost_eat_streak = 0

    def next_level(self) -> None:
        self.level += 1
        self.level_complete = False
        self._place_pellets()
        self._reset_positions()

    # ------------------------------------------------------------------
    # Exposition de l'état
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
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
        grid = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if (x, y) in self.walls:
                    row.append('W')
                elif (x, y) in self.ghost_walls:
                    row.append('D')
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
