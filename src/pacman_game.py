#!/usr/bin/env python3
"""
Pac-Man playable game loaded from maze.json.

Controls:
  Arrow keys / WASD  — move Pac-Man
  R                  — restart
  Q / Escape         — quit

Design:
  GameState  — holds all mutable data (pellets, energizers, pac-man).
               Designed to be extended with ghosts: add self.ghosts here.
  GameRenderer — pure drawing, no logic.

Pellet placement rules (all PATH nodes except):
  - Ghost-house surround zone  x ∈ [9,18], y ∈ [10,16]
  - Tunnel edge columns        x == 0  or  x == width-1
  - Pac-Man spawn              (13, 23)

4 energizers are placed near the four corners of the playable area
(nearest pellet-eligible tile to each of the 4 anchor positions).
"""

import argparse
import json
import math
import sys
from typing import Dict, List, Optional, Set, Tuple

try:
    import pygame
except ImportError:
    sys.exit("pygame is required:  pip install pygame")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TILE: int = 20          # pixels per tile (overridable via CLI)
FPS:  int = 60
PACMAN_SPEED: float = 5.0   # tiles per second

# Ghost-house surround zone — no pellets (matches pacman_map.py exclusion)
GHOST_ZONE_X: Tuple[int, int] = (9, 18)
GHOST_ZONE_Y: Tuple[int, int] = (10, 16)

PACMAN_START: Tuple[int, int] = (13, 23)

# Energizer anchor corners — best 4 pellet tiles nearest these positions
ENERGIZER_ANCHORS: List[Tuple[int, int]] = [(1, 3), (26, 3), (1, 23), (26, 23)]

POWER_UP_DURATION: float = 8.0   # seconds

# Directions
DIR_NONE:  Tuple[int, int] = ( 0,  0)
DIR_UP:    Tuple[int, int] = ( 0, -1)
DIR_DOWN:  Tuple[int, int] = ( 0,  1)
DIR_LEFT:  Tuple[int, int] = (-1,  0)
DIR_RIGHT: Tuple[int, int] = ( 1,  0)

# Colours
BLACK         = (  0,   0,   0)
BLUE          = ( 33,  33, 222)
WHITE         = (255, 255, 255)
YELLOW        = (255, 255,   0)
PELLET_COL    = (255, 255, 184)
ENERGY_COL    = (255, 255, 255)
GHOST_WALL_COL= (255, 182, 193)
POWER_UP_COL  = (  0, 200, 255)

KEY_TO_DIR = {}   # populated after pygame.init()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nearest_pellet(
    anchor: Tuple[int, int],
    candidates: List[Tuple[int, int]],
) -> Tuple[int, int]:
    ax, ay = anchor
    return min(candidates, key=lambda p: (p[0] - ax) ** 2 + (p[1] - ay) ** 2)


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

class GameState:
    """
    Holds all mutable game data.

    To add ghosts later:
      - Add  self.ghosts: List[GhostState]  in __init__ / reset()
      - Call ghost.update(dt, self) in update()
    """

    def __init__(self, maze_data: Dict):
        meta = maze_data["metadata"]
        self.width:  int = meta["width"]
        self.height: int = meta["height"]

        # Build walkable tile set
        self.walkable: Set[Tuple[int, int]] = set()
        for node in maze_data["nodes"]:
            if node["type"] != "GHOSTSPACE":
                self.walkable.add((node["x"], node["y"]))

        # Adjacency map (edges already include tunnel connections from generate.py)
        self.neighbors: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {
            p: set() for p in self.walkable
        }
        for edge in maze_data["edges"]:
            a: Tuple[int, int] = tuple(int(v) for v in edge["from"].split(","))  # type: ignore
            b: Tuple[int, int] = tuple(int(v) for v in edge["to"].split(","))    # type: ignore
            if a in self.neighbors:
                self.neighbors[a].add(b)
            if b in self.neighbors:
                self.neighbors[b].add(a)

        # Compute pellet-eligible tiles once
        pellet_candidates: List[Tuple[int, int]] = []
        for (x, y) in self.walkable:
            in_ghost_zone = (
                GHOST_ZONE_X[0] <= x <= GHOST_ZONE_X[1]
                and GHOST_ZONE_Y[0] <= y <= GHOST_ZONE_Y[1]
            )
            is_tunnel = x == 0 or x == self.width - 1
            is_spawn  = (x, y) == PACMAN_START
            if not in_ghost_zone and not is_tunnel and not is_spawn:
                pellet_candidates.append((x, y))

        # Place 4 energizers near the 4 corner anchors (greedy nearest)
        self.energizers: Set[Tuple[int, int]] = set()
        remaining = list(pellet_candidates)
        for anchor in ENERGIZER_ANCHORS:
            chosen = _nearest_pellet(anchor, remaining)
            self.energizers.add(chosen)
            remaining.remove(chosen)

        # All other candidates are normal pellets
        self.pellets: Set[Tuple[int, int]] = set(remaining)

        self.score: int = 0
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        """Restore full game to initial state."""
        self.pellets_remaining: Set[Tuple[int, int]]    = set(self.pellets)
        self.energizers_remaining: Set[Tuple[int, int]] = set(self.energizers)

        # Pac-Man sub-tile position (float tiles)
        self.pacman_px: float = float(PACMAN_START[0])
        self.pacman_py: float = float(PACMAN_START[1])
        self.pacman_tile: Tuple[int, int] = PACMAN_START
        self.pacman_dir: Tuple[int, int] = DIR_NONE
        self.pacman_next_dir: Tuple[int, int] = DIR_NONE
        self.pacman_moving: bool = False

        # Animation
        self.mouth_phase: float = 0.0   # radians, drives mouth open/close

        # State flags
        self.score = 0
        self.won:       bool  = False
        self.powered_up: bool = False
        self.power_timer: float = 0.0

    # ------------------------------------------------------------------
    def _can_move(
        self, tile: Tuple[int, int], direction: Tuple[int, int]
    ) -> bool:
        """True when moving in direction from tile lands on a walkable tile."""
        neighbor: Tuple[int, int] = (
            tile[0] + direction[0],
            tile[1] + direction[1],
        )
        return neighbor in self.neighbors.get(tile, set())

    # ------------------------------------------------------------------
    def update(self, dt: float, requested_dir: Tuple[int, int]):
        """
        Advance the game by dt seconds.

        requested_dir  — direction currently held by the player (DIR_NONE if none).
        Subclasses / callers can pass ghost updates separately after calling super().
        """
        if self.won:
            return

        # Buffer direction request
        if requested_dir != DIR_NONE:
            self.pacman_next_dir = requested_dir

        # Check whether Pac-Man is close enough to a tile centre to steer
        tx = int(round(self.pacman_px))
        ty = int(round(self.pacman_py))
        near_centre = abs(self.pacman_px - tx) < 0.15 and abs(self.pacman_py - ty) < 0.15

        if near_centre:
            self.pacman_tile = (tx, ty)
            # Snap to exact centre to avoid floating-point drift
            self.pacman_px = float(tx)
            self.pacman_py = float(ty)

            # Try buffered direction first, then current direction
            self.pacman_moving = False
            for d in (self.pacman_next_dir, self.pacman_dir):
                if d != DIR_NONE and self._can_move(self.pacman_tile, d):
                    if d == self.pacman_next_dir:
                        self.pacman_next_dir = DIR_NONE
                    self.pacman_dir = d
                    self.pacman_moving = True
                    break

        # Move
        if self.pacman_moving and self.pacman_dir != DIR_NONE:
            self.pacman_px += self.pacman_dir[0] * PACMAN_SPEED * dt
            self.pacman_py += self.pacman_dir[1] * PACMAN_SPEED * dt

        # Animate mouth
        if self.pacman_moving:
            self.mouth_phase = (self.mouth_phase + dt * 6.0) % (2 * math.pi)
        else:
            self.mouth_phase = 0.0

        # Power-up timer
        if self.powered_up:
            self.power_timer -= dt
            if self.power_timer <= 0.0:
                self.powered_up = False

        # Eat pellets / energizers at current tile
        tile: Tuple[int, int] = (int(round(self.pacman_px)), int(round(self.pacman_py)))
        if tile in self.pellets_remaining:
            self.pellets_remaining.discard(tile)
            self.score += 10
        if tile in self.energizers_remaining:
            self.energizers_remaining.discard(tile)
            self.score += 50
            self.powered_up = True
            self.power_timer = POWER_UP_DURATION

        # Win condition
        if not self.pellets_remaining and not self.energizers_remaining:
            self.won = True


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class GameRenderer:
    """
    Pure rendering — no game logic.
    Draws walls, ghost walls, pellets, energizers, Pac-Man and HUD.

    To add ghost rendering later: call draw_ghost(surface, ghost_state) from draw().
    """

    def __init__(self, maze_data: Dict, tile: int = TILE):
        self.t = tile
        self.meta = maze_data["metadata"]

        # Pre-build wall rect list for fast drawing
        self.wall_rects: List[pygame.Rect] = [
            pygame.Rect(w["x"] * tile, w["y"] * tile, tile, tile)
            for w in maze_data["walls"]
        ]
        self.ghost_wall_tiles: List[Tuple[int, int]] = [
            (gw["x"], gw["y"]) for gw in maze_data["ghost_walls"]
        ]

    @property
    def screen_size(self) -> Tuple[int, int]:
        return (self.meta["width"] * self.t, self.meta["height"] * self.t + 40)

    # ------------------------------------------------------------------
    def draw(self, surface: "pygame.Surface", state: GameState):
        t = self.t
        surface.fill(BLACK)

        # Walls
        for rect in self.wall_rects:
            pygame.draw.rect(surface, BLUE, rect)

        # Ghost-house door (horizontal pink bar)
        for gx, gy in self.ghost_wall_tiles:
            mid_y = gy * t + t // 2
            pygame.draw.rect(surface, GHOST_WALL_COL, (gx * t, mid_y - 1, t, 3))

        # Pellets — small warm-yellow dot
        pr = max(1, t // 10)
        for (px, py) in state.pellets_remaining:
            pygame.draw.circle(
                surface, PELLET_COL,
                (px * t + t // 2, py * t + t // 2), pr,
            )

        # Energizers — large white pulsing dot
        pulse = int(abs(math.sin(pygame.time.get_ticks() / 300.0)) * 3)
        er = max(3, t // 5) + pulse
        for (ex, ey) in state.energizers_remaining:
            pygame.draw.circle(
                surface, ENERGY_COL,
                (ex * t + t // 2, ey * t + t // 2), er,
            )

        # Pac-Man
        self._draw_pacman(surface, state)

        # HUD
        self._draw_hud(surface, state)

    # ------------------------------------------------------------------
    def _draw_pacman(self, surface: "pygame.Surface", state: GameState):
        t = self.t
        cx = int(state.pacman_px * t + t // 2)
        cy = int(state.pacman_py * t + t // 2)
        r  = max(4, t // 2 - 2)

        mouth_deg = int(abs(math.sin(state.mouth_phase)) * 35)

        # Direction → rotation so the mouth faces movement direction
        dir_rot = {
            DIR_RIGHT: 0,
            DIR_LEFT:  180,
            DIR_UP:    90,
            DIR_DOWN:  270,
            DIR_NONE:  0,
        }
        rot = dir_rot.get(state.pacman_dir, 0)

        start_a = math.radians(mouth_deg + rot)
        end_a   = math.radians(360 - mouth_deg + rot)

        points = [(cx, cy)]
        segs = 40
        for i in range(segs + 1):
            a = start_a + (end_a - start_a) * i / segs
            points.append((
                cx + int(r * math.cos(a)),
                cy - int(r * math.sin(a)),
            ))
        if len(points) > 2:
            pygame.draw.polygon(surface, YELLOW, points)

    # ------------------------------------------------------------------
    def _draw_hud(self, surface: "pygame.Surface", state: GameState):
        w = self.meta["width"] * self.t
        h = self.meta["height"] * self.t
        font = pygame.font.SysFont("monospace", 18, bold=True)

        score_surf = font.render(f"SCORE  {state.score:>6}", True, WHITE)
        surface.blit(score_surf, (10, h + 8))

        if state.won:
            win_surf = font.render("YOU WIN!   R = restart", True, YELLOW)
            surface.blit(win_surf, (w // 2 - 110, h + 8))

        if state.powered_up:
            p_surf = font.render(
                f"POWERED  {state.power_timer:.1f}s", True, POWER_UP_COL
            )
            surface.blit(p_surf, (w - 190, h + 8))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    global KEY_TO_DIR

    parser = argparse.ArgumentParser(
        description="Play Pac-Man on a generated maze (arrow keys / WASD, R = restart, Q = quit)",
    )
    parser.add_argument("maze_json", help="Path to maze.json")
    parser.add_argument(
        "-s", "--size", type=int, default=TILE,
        help=f"Tile size in pixels (default: {TILE})",
    )
    args = parser.parse_args()

    with open(args.maze_json) as f:
        maze_data = json.load(f)

    pygame.init()

    KEY_TO_DIR = {
        pygame.K_UP:    DIR_UP,
        pygame.K_w:     DIR_UP,
        pygame.K_DOWN:  DIR_DOWN,
        pygame.K_s:     DIR_DOWN,
        pygame.K_LEFT:  DIR_LEFT,
        pygame.K_a:     DIR_LEFT,
        pygame.K_RIGHT: DIR_RIGHT,
        pygame.K_d:     DIR_RIGHT,
    }

    state    = GameState(maze_data)
    renderer = GameRenderer(maze_data, tile=args.size)
    screen   = pygame.display.set_mode(renderer.screen_size)
    pygame.display.set_caption("Pac-Man")
    clock    = pygame.time.Clock()

    requested_dir: Tuple[int, int] = DIR_NONE

    while True:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return 0
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    return 0
                if event.key == pygame.K_r:
                    state.reset()
                    requested_dir = DIR_NONE
                d = KEY_TO_DIR.get(event.key)
                if d is not None:
                    requested_dir = d
            if event.type == pygame.KEYUP:
                d = KEY_TO_DIR.get(event.key)
                if d is not None and d == requested_dir:
                    requested_dir = DIR_NONE

        state.update(dt, requested_dir)
        renderer.draw(screen, state)
        pygame.display.flip()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
