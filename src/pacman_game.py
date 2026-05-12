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

# Ghost colours
GHOST_FRIGHT_COL  = (  0,   0, 200)
GHOST_FLASH_COL   = (255, 255, 255)
GHOST_EYE_WHITE   = (255, 255, 255)
GHOST_EYE_IRIS    = (  0,   0, 200)

# Ghost modes
GHOST_WAITING    = "waiting"      # pre-release, frozen at start tile
GHOST_EXITING    = "exiting"      # released — navigating to corridor exit
GHOST_SCATTER    = "scatter"      # retreat to assigned corner
GHOST_CHASE      = "chase"        # use targeting algorithm
GHOST_FRIGHTENED = "frightened"   # random turns, blue
GHOST_DEAD       = "dead"         # eaten — brief pause then respawn

# Tile the ghost must reach when exiting the house (centre of the y=11 corridor)
GHOST_EXIT_TILE: Tuple[int, int] = (13, 11)

# Global scatter/chase schedule [(mode, duration_s)] — last entry repeats
GHOST_MODE_SCHEDULE: List[Tuple[str, float]] = [
    (GHOST_SCATTER,  7.0),
    (GHOST_CHASE,   20.0),
    (GHOST_SCATTER,  7.0),
    (GHOST_CHASE,   20.0),
    (GHOST_SCATTER,  5.0),
    (GHOST_CHASE,   20.0),
    (GHOST_SCATTER,  5.0),
    (GHOST_CHASE,  9999.0),
]

GHOST_SPEED:          float = 4.5   # tiles / sec
GHOST_FRIGHT_SPEED:   float = 2.5
GHOST_DEAD_DURATION:  float = 3.0   # seconds before respawn
GHOST_FRIGHT_FLASH_T: float = 2.0   # seconds remaining when flashing starts

GHOST_EAT_SCORES: List[int] = [200, 400, 800, 1600]  # cumulative per power pellet

# Ghost house bounding-box corners (as given by maze design)
# Walls are 1-tile thick, so the interior space is 1 tile inside on all sides.
GHOST_HOUSE_TL: Tuple[int, int] = (10, 11)   # top-left  corner
GHOST_HOUSE_BR: Tuple[int, int] = (17, 15)   # bottom-right corner
# Interior: x = TL.x+1 .. BR.x-1,  y = TL.y+1 .. BR.y-1  →  x=11..16, y=12..14

# Per-ghost config: (name, colour, scatter_corner, release_delay_s, start_tile)
#   Blinky  (13,11)  on the y=11 corridor above the door — exits immediately
#   Pinky   (13,13)  interior centre — patrols up/down before release
#   Inky    (11,13)  interior left   — patrols up/down before release
#   Clyde   (16,13)  interior right  — patrols up/down before release
GHOST_CONFIGS: List[Tuple[str, Tuple[int,int,int], Tuple[int,int], float, Tuple[int,int]]] = [
    ("blinky", (255,   0,   0), (25,  0),  0.0, (13, 11)),
    ("pinky",  (255, 184, 255), ( 2,  0),  3.0, (13, 13)),
    ("inky",   (  0, 255, 255), (27, 30),  6.0, (11, 13)),
    ("clyde",  (255, 184,  82), ( 0, 30),  9.0, (16, 13)),
]

PACMAN_LIVES: int = 3

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
# Ghost
# ---------------------------------------------------------------------------

class GhostState:
    """
    One ghost.  Uses the same tile-progress movement model as Pac-Man.

    Targeting algorithms (reference: Pac-Man Dossier):
      blinky — direct chase: Pac-Man's current tile
      pinky  — 4-tile ambush ahead of Pac-Man (with classic UP overflow bug)
      inky   — flanking: 2× vector from Blinky to 2 tiles ahead of Pac-Man
      clyde  — shy: chases Pac-Man if ≥8 tiles away, else retreats to corner

    Direction priority at intersections: UP > LEFT > DOWN > RIGHT (never reverse).
    Frightened: random valid direction, half speed.
    """

    def __init__(
        self,
        name: str,
        color: Tuple[int, int, int],
        scatter_corner: Tuple[int, int],
        release_delay: float,
        start_tile: Tuple[int, int],
    ):
        self.name           = name
        self.color          = color
        self.scatter_corner = scatter_corner
        self.release_delay  = release_delay
        self.start_tile     = start_tile
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.tile:           Tuple[int, int] = self.start_tile
        self.target:         Tuple[int, int] = self.start_tile
        self.progress:       float           = 0.0
        self.dir:            Tuple[int, int] = DIR_NONE
        self.moving:         bool            = False

        self.mode:           str             = GHOST_WAITING
        self.wait_timer:     float           = self.release_delay
        self.fright_timer:   float           = 0.0
        self.dead_timer:     float           = 0.0
        # Patrol direction used while waiting inside the ghost house (left/right)
        self.patrol_dir:     Tuple[int, int] = DIR_LEFT

    # ------------------------------------------------------------------
    @property
    def px(self) -> float:
        return self.tile[0] + (self.target[0] - self.tile[0]) * self.progress

    @property
    def py(self) -> float:
        return self.tile[1] + (self.target[1] - self.tile[1]) * self.progress

    # ------------------------------------------------------------------
    def frighten(self, duration: float):
        """Enter frightened mode (power pellet eaten by Pac-Man)."""
        if self.mode in (GHOST_WAITING, GHOST_EXITING, GHOST_DEAD):
            return
        # Force reversal — swap tile↔target and invert progress
        if self.moving and self.dir != DIR_NONE:
            self.tile, self.target = self.target, self.tile
            self.progress = 1.0 - self.progress
            self.dir = (-self.dir[0], -self.dir[1])
        self.mode         = GHOST_FRIGHTENED
        self.fright_timer = duration

    def eat(self):
        """Ghost was eaten while frightened — short pause then respawn."""
        self.mode       = GHOST_DEAD
        self.dead_timer = GHOST_DEAD_DURATION
        self.moving     = False

    # ------------------------------------------------------------------
    def update(self, dt: float, state: "GameState"):
        if self.mode == GHOST_WAITING:
            self.wait_timer -= dt
            # Patrol left/right inside the house while counting down
            if self.moving:
                self.progress += GHOST_FRIGHT_SPEED * dt
                if self.progress >= 1.0:
                    self.tile     = self.target
                    self.progress = 0.0
                    self._patrol(state)
            else:
                self._patrol(state)
            # Release: enter exiting mode to navigate up through the door
            if self.wait_timer <= 0.0:
                self.mode = GHOST_EXITING
                self._exit_house(state)
            return

        if self.mode == GHOST_EXITING:
            speed = GHOST_SPEED
            if self.moving:
                self.progress += speed * dt
                if self.progress >= 1.0:
                    self.tile     = self.target
                    self.progress = 0.0
                    # Reached corridor — start normal scatter/chase
                    if self.tile[1] <= GHOST_EXIT_TILE[1]:
                        self.mode = state.ghost_mode
                        self._start_moving(state)
                    else:
                        self._exit_house(state)
            else:
                self._exit_house(state)
            return

        if self.mode == GHOST_DEAD:
            self.dead_timer -= dt
            if self.dead_timer <= 0.0:
                self.reset()
                self.mode       = GHOST_EXITING
                self.wait_timer = 0.0
                self._exit_house(state)
            return

        if self.mode == GHOST_FRIGHTENED:
            self.fright_timer -= dt
            if self.fright_timer <= 0.0:
                self.mode = state.ghost_mode

        speed = GHOST_FRIGHT_SPEED if self.mode == GHOST_FRIGHTENED else GHOST_SPEED

        if self.moving:
            self.progress += speed * dt
            if self.progress >= 1.0:
                self.tile     = self.target
                self.progress = 0.0
                self._pick_next(state)
        else:
            self._start_moving(state)

    # ------------------------------------------------------------------
    def _patrol(self, state: "GameState"):
        """Bounce left/right within the ghost house interior while waiting."""
        for attempt in range(2):
            nb = state._target_tile(self.tile, self.patrol_dir)
            if (state._can_move(self.tile, self.patrol_dir, ghost=True)
                    and nb in state.ghost_interior):
                self.dir      = self.patrol_dir
                self.target   = nb
                self.progress = 0.0
                self.moving   = True
                return
            # Hit boundary — reverse
            self.patrol_dir = (-self.patrol_dir[0], -self.patrol_dir[1])
        self.moving = False

    def _exit_house(self, state: "GameState"):
        """Navigate step-by-step toward GHOST_EXIT_TILE using ghost_neighbors."""
        d = self._choose_dir(self.tile, GHOST_EXIT_TILE, DIR_NONE, state, ghost_nav=True)
        if d != DIR_NONE:
            self._commit(d, state)
        else:
            self.moving = False

    def _start_moving(self, state: "GameState"):
        """Choose first direction and begin moving."""
        t  = self._get_target(state)
        d  = self._choose_dir(self.tile, t, DIR_NONE, state)
        if d != DIR_NONE:
            self._commit(d, state)

    def _pick_next(self, state: "GameState"):
        """Called on tile arrival — choose the next direction."""
        import random
        reverse = (-self.dir[0], -self.dir[1]) if self.dir != DIR_NONE else DIR_NONE

        if self.mode == GHOST_FRIGHTENED:
            options = [
                d for d in (DIR_UP, DIR_LEFT, DIR_DOWN, DIR_RIGHT)
                if d != reverse and state._can_move(self.tile, d)
            ]
            if not options and reverse != DIR_NONE and state._can_move(self.tile, reverse):
                options = [reverse]
            if options:
                self._commit(random.choice(options), state)
            else:
                self.moving = False
        else:
            target = self._get_target(state)
            d      = self._choose_dir(self.tile, target, reverse, state)
            if d == DIR_NONE and reverse != DIR_NONE and state._can_move(self.tile, reverse):
                d = reverse          # dead end — reverse
            if d != DIR_NONE:
                self._commit(d, state)
            else:
                self.moving = False

    def _commit(self, direction: Tuple[int, int], state: "GameState"):
        self.dir      = direction
        self.target   = state._target_tile(self.tile, direction)
        self.progress = 0.0
        self.moving   = True

    # ------------------------------------------------------------------
    def _choose_dir(
        self,
        from_tile: Tuple[int, int],
        target:    Tuple[int, int],
        forbidden: Tuple[int, int],
        state:     "GameState",
        ghost_nav: bool = False,
    ) -> Tuple[int, int]:
        """
        Classic ghost direction selection.
        Priority when distances tie: UP > LEFT > DOWN > RIGHT.
        Never choose the forbidden (reverse) direction.
        ghost_nav=True uses ghost_neighbors (for exiting the house).
        """
        best_dir  = DIR_NONE
        best_dist = float("inf")
        for d in (DIR_UP, DIR_LEFT, DIR_DOWN, DIR_RIGHT):
            if d == forbidden:
                continue
            if not state._can_move(from_tile, d, ghost=ghost_nav):
                continue
            nb   = state._target_tile(from_tile, d)
            dist = (nb[0] - target[0]) ** 2 + (nb[1] - target[1]) ** 2
            if dist < best_dist:
                best_dist = dist
                best_dir  = d
        return best_dir

    # ------------------------------------------------------------------
    def _get_target(self, state: "GameState") -> Tuple[int, int]:
        """Target tile for scatter or chase mode."""
        if self.mode == GHOST_SCATTER:
            return self.scatter_corner

        # ---- CHASE ----
        px, py   = state.pacman_tile
        pdx, pdy = state.pacman_dir if state.pacman_dir != DIR_NONE else (0, 0)

        if self.name == "blinky":
            # Direct chase: Pac-Man's tile
            return (px, py)

        elif self.name == "pinky":
            # 4 tiles ahead of Pac-Man (+ classic UP overflow)
            tx = px + 4 * pdx
            ty = py + 4 * pdy
            if state.pacman_dir == DIR_UP:
                tx -= 4
            return (tx, ty)

        elif self.name == "inky":
            # Pivot 2 tiles ahead, doubled from Blinky's position
            tx = px + 2 * pdx
            ty = py + 2 * pdy
            if state.pacman_dir == DIR_UP:
                tx -= 2
            blinky = next((g for g in state.ghosts if g.name == "blinky"), None)
            bx, by = blinky.tile if blinky else (0, 0)
            return (bx + 2 * (tx - bx), by + 2 * (ty - by))

        else:  # clyde
            # Target Pac-Man when ≥8 tiles away, else retreat to scatter corner
            dx = px - self.tile[0]
            dy = py - self.tile[1]
            if dx * dx + dy * dy >= 64:
                return (px, py)
            return self.scatter_corner

class GameState:
    """All mutable game data: maze topology, Pac-Man, ghosts, score, lives."""

    def __init__(self, maze_data: Dict):
        meta = maze_data["metadata"]
        self.width:  int = meta["width"]
        self.height: int = meta["height"]

        # Build walkable tile set
        self.walkable: Set[Tuple[int, int]] = set()
        for node in maze_data["nodes"]:
            if node["type"] != "GHOSTSPACE":
                self.walkable.add((node["x"], node["y"]))

        # Adjacency map (lateral edges; wrap-around handled by tunnel_rows)
        self.neighbors: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {
            p: set() for p in self.walkable
        }
        for edge in maze_data["edges"]:
            a: Tuple[int, int] = tuple(int(v) for v in edge["from"].split(","))  # type: ignore
            b: Tuple[int, int] = tuple(int(v) for v in edge["to"].split(","))    # type: ignore
            # Only register the edge when both endpoints are walkable tiles;
            # edges that touch wall/GHOSTSPACE nodes are discarded so ghosts
            # (and Pac-Man) cannot navigate into non-walkable cells.
            if a in self.neighbors and b in self.neighbors:
                self.neighbors[a].add(b)
                self.neighbors[b].add(a)

        # Rows where both the left edge (x=0) and right edge (x=width-1) are walkable
        # — these are horizontal tunnel passages.
        left_ys  = {y for (x, y) in self.walkable if x == 0}
        right_ys = {y for (x, y) in self.walkable if x == self.width - 1}
        self.tunnel_rows: Set[int] = left_ys & right_ys

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

        # ---- Ghost house interior -----------------------------------------------
        # These tiles are solid walls for Pac-Man but freely navigable by ghosts.
        tl_x, tl_y = GHOST_HOUSE_TL
        br_x, br_y = GHOST_HOUSE_BR
        self.ghost_interior: FrozenSet[Tuple[int, int]] = frozenset(
            (x, y)
            for x in range(tl_x + 1, br_x)   # x = 11..16
            for y in range(tl_y + 1, br_y)    # y = 12..14
        )
        # Ghost-only adjacency = normal neighbours + interior grid + corridor edges
        ghost_all = self.walkable | self.ghost_interior
        self.ghost_neighbors: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {
            p: set(self.neighbors.get(p, set())) for p in ghost_all
        }
        for (x, y) in self.ghost_interior:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (x + dx, y + dy)
                if nb in ghost_all:
                    self.ghost_neighbors[(x, y)].add(nb)
                    if nb not in self.ghost_neighbors:
                        self.ghost_neighbors[nb] = set()
                    self.ghost_neighbors[nb].add((x, y))

        # Build ghost objects (topology-independent — reset() sets their state)
        self.ghosts: List[GhostState] = [
            GhostState(name, color, corner, delay, start)
            for name, color, corner, delay, start in GHOST_CONFIGS
        ]

        self.score: int = 0
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        """Restore full game to initial state."""
        self.pellets_remaining: Set[Tuple[int, int]]    = set(self.pellets)
        self.energizers_remaining: Set[Tuple[int, int]] = set(self.energizers)

        # Tile-progress movement model:
        #   pacman_tile     — tile Pac-Man is currently leaving (or resting on)
        #   pacman_target   — tile Pac-Man is moving toward (== pacman_tile if stopped)
        #   pacman_progress — 0.0 (at pacman_tile) → 1.0 (arrived at pacman_target)
        self.pacman_tile:     Tuple[int, int] = PACMAN_START
        self.pacman_target:   Tuple[int, int] = PACMAN_START
        self.pacman_progress: float           = 0.0
        self.pacman_dir:            Tuple[int, int] = DIR_NONE
        self.pacman_next_dir:      Tuple[int, int] = DIR_NONE
        self.pacman_moving:        bool            = False
        # True while Pac-Man is sliding through a tunnel (visual: off-screen exit)
        self.pacman_tunnel_transit: bool           = False

        # Animation
        self.mouth_phase: float = 0.0

        # State flags
        self.score       = 0
        self.lives:       int   = PACMAN_LIVES
        self.won:         bool  = False
        self.game_over:   bool  = False
        self.powered_up:  bool  = False
        self.power_timer: float = 0.0

        # Ghost global mode cycle
        self.ghost_mode:       str   = GHOST_SCATTER
        self.ghost_mode_idx:   int   = 0
        self.ghost_mode_timer: float = GHOST_MODE_SCHEDULE[0][1]
        self.ghosts_eaten:     int   = 0

        for ghost in self.ghosts:
            ghost.reset()

    # ------------------------------------------------------------------
    @property
    def pacman_px(self) -> float:
        """Interpolated x position in tile coordinates.

        During a tunnel transit the position slides *off-screen* in the
        direction of travel rather than jumping to the far side.
        """
        if self.pacman_tunnel_transit:
            # Slide off the edge: move further in the current direction
            return self.pacman_tile[0] + self.pacman_dir[0] * self.pacman_progress
        return self.pacman_tile[0] + (self.pacman_target[0] - self.pacman_tile[0]) * self.pacman_progress

    @property
    def pacman_py(self) -> float:
        """Interpolated y position in tile coordinates."""
        return self.pacman_tile[1] + (self.pacman_target[1] - self.pacman_tile[1]) * self.pacman_progress

    # ------------------------------------------------------------------
    def _can_move(
        self, tile: Tuple[int, int], direction: Tuple[int, int], ghost: bool = False
    ) -> bool:
        """True when moving in *direction* from *tile* is permitted.

        ghost=True uses the ghost-only adjacency map which includes the
        house interior; ghost=False (default) uses the Pac-Man graph.
        """
        nbrs = self.ghost_neighbors if ghost else self.neighbors
        neighbor: Tuple[int, int] = (
            tile[0] + direction[0],
            tile[1] + direction[1],
        )
        if neighbor in nbrs.get(tile, set()):
            return True
        # Tunnel wrap: left edge moving left, or right edge moving right
        if direction == (-1, 0) and tile[0] == 0 and tile[1] in self.tunnel_rows:
            return True
        if direction == (1, 0) and tile[0] == self.width - 1 and tile[1] in self.tunnel_rows:
            return True
        return False

    def _target_tile(
        self, tile: Tuple[int, int], direction: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Return the tile Pac-Man lands on when moving in direction from tile.

        Handles tunnel wrap-around: exiting the left edge arrives at the
        right edge and vice versa.
        """
        if direction == (-1, 0) and tile[0] == 0 and tile[1] in self.tunnel_rows:
            return (self.width - 1, tile[1])
        if direction == (1, 0) and tile[0] == self.width - 1 and tile[1] in self.tunnel_rows:
            return (0, tile[1])
        return (tile[0] + direction[0], tile[1] + direction[1])

    def _is_tunnel_move(
        self, tile: Tuple[int, int], direction: Tuple[int, int]
    ) -> bool:
        """True when the move exits through a tunnel."""
        return (
            direction == (-1, 0) and tile[0] == 0 and tile[1] in self.tunnel_rows
        ) or (
            direction == (1, 0) and tile[0] == self.width - 1 and tile[1] in self.tunnel_rows
        )

    # ------------------------------------------------------------------
    def update(self, dt: float, requested_dir: Tuple[int, int]):
        """
        Advance the game by dt seconds.

        Uses a tile-progress model: Pac-Man slides from pacman_tile to
        pacman_target over (1 / PACMAN_SPEED) seconds.  Direction changes
        are applied only at tile boundaries — no floating-point drift.

        requested_dir — direction currently held by the player.
        """
        if self.won or self.game_over:
            return

        # Buffer direction request
        if requested_dir != DIR_NONE:
            self.pacman_next_dir = requested_dir

        if self.pacman_moving:
            # Advance progress
            self.pacman_progress += PACMAN_SPEED * dt

            if self.pacman_progress >= 1.0:
                # Arrived at target tile
                self.pacman_tile            = self.pacman_target
                self.pacman_progress        = 0.0
                self.pacman_tunnel_transit  = False

                # Eat pellet / energizer on arrival
                self._eat(self.pacman_tile)

                # Try buffered direction, then current direction
                moved = False
                for d in (self.pacman_next_dir, self.pacman_dir):
                    if d != DIR_NONE and self._can_move(self.pacman_tile, d):
                        if d == self.pacman_next_dir:
                            self.pacman_next_dir = DIR_NONE
                        self.pacman_dir            = d
                        self.pacman_target         = self._target_tile(self.pacman_tile, d)
                        self.pacman_progress       = 0.0
                        self.pacman_tunnel_transit = self._is_tunnel_move(self.pacman_tile, d)
                        moved = True
                        break

                if not moved:
                    self.pacman_moving = False
                    self.pacman_target = self.pacman_tile

        else:
            # Stopped — try to start moving with buffered or current direction
            for d in (self.pacman_next_dir, self.pacman_dir):
                if d != DIR_NONE and self._can_move(self.pacman_tile, d):
                    if d == self.pacman_next_dir:
                        self.pacman_next_dir = DIR_NONE
                    self.pacman_dir            = d
                    self.pacman_target         = self._target_tile(self.pacman_tile, d)
                    self.pacman_progress       = 0.0
                    self.pacman_tunnel_transit = self._is_tunnel_move(self.pacman_tile, d)
                    self.pacman_moving         = True
                    break

        # Mouth animation
        if self.pacman_moving:
            self.mouth_phase = (self.mouth_phase + dt * 6.0) % (2 * math.pi)
        else:
            self.mouth_phase = 0.0

        # Power-up timer
        if self.powered_up:
            self.power_timer -= dt
            if self.power_timer <= 0.0:
                self.powered_up = False

        # ---- Ghost global mode cycle (paused while any ghost is frightened) ----
        if not any(g.mode == GHOST_FRIGHTENED for g in self.ghosts):
            self.ghost_mode_timer -= dt
            if self.ghost_mode_timer <= 0.0:
                self.ghost_mode_idx = min(
                    self.ghost_mode_idx + 1, len(GHOST_MODE_SCHEDULE) - 1
                )
                self.ghost_mode, duration = GHOST_MODE_SCHEDULE[self.ghost_mode_idx]
                self.ghost_mode_timer = duration
                for g in self.ghosts:
                    if g.mode not in (GHOST_WAITING, GHOST_EXITING, GHOST_DEAD, GHOST_FRIGHTENED):
                        g.mode = self.ghost_mode

        # ---- Update ghosts ----
        for ghost in self.ghosts:
            ghost.update(dt, self)

        # ---- Pac-Man vs ghost collision ----
        for ghost in self.ghosts:
            if ghost.mode in (GHOST_WAITING, GHOST_EXITING, GHOST_DEAD):
                continue
            if ghost.tile != self.pacman_tile:
                continue
            if ghost.mode == GHOST_FRIGHTENED:
                ghost.eat()
                self.score       += GHOST_EAT_SCORES[min(self.ghosts_eaten, 3)]
                self.ghosts_eaten += 1
            else:
                self.lives -= 1
                if self.lives <= 0:
                    self.game_over = True
                else:
                    self._respawn_pacman()
                break

        # Win condition
        if not self.pellets_remaining and not self.energizers_remaining:
            self.won = True

    # ------------------------------------------------------------------
    def _eat(self, tile: Tuple[int, int]):
        """Consume any pellet or energizer on tile."""
        if tile in self.pellets_remaining:
            self.pellets_remaining.discard(tile)
            self.score += 10
        if tile in self.energizers_remaining:
            self.energizers_remaining.discard(tile)
            self.score += 50
            self.powered_up   = True
            self.power_timer  = POWER_UP_DURATION
            self.ghosts_eaten = 0
            for ghost in self.ghosts:
                ghost.frighten(POWER_UP_DURATION)

    # ------------------------------------------------------------------
    def _respawn_pacman(self):
        """Reset Pac-Man to spawn without clearing pellets (lose a life)."""
        self.pacman_tile           = PACMAN_START
        self.pacman_target         = PACMAN_START
        self.pacman_progress       = 0.0
        self.pacman_dir            = DIR_NONE
        self.pacman_next_dir       = DIR_NONE
        self.pacman_moving         = False
        self.pacman_tunnel_transit = False
        self.mouth_phase           = 0.0
        for ghost in self.ghosts:
            ghost.reset()


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

        # Ghosts
        for ghost in state.ghosts:
            self._draw_ghost(surface, ghost)

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
    def _draw_ghost(self, surface: "pygame.Surface", ghost: GhostState):
        """Draw a single ghost in its current mode/colour."""
        t = self.t

        if ghost.mode == GHOST_WAITING:
            # Faint silhouette so the player sees where they will spawn
            cx = ghost.tile[0] * t + t // 2
            cy = ghost.tile[1] * t + t // 2
            r  = max(4, t // 2 - 2)
            dim = tuple(max(0, c - 160) for c in ghost.color)
            self._ghost_body(surface, cx, cy, r, dim)
            return

        # GHOST_EXITING — full colour, eyes pointing upward toward door
        if ghost.mode == GHOST_EXITING:
            cx = int(ghost.px * t + t // 2)
            cy = int(ghost.py * t + t // 2)
            r  = max(4, t // 2 - 2)
            self._ghost_body(surface, cx, cy, r, ghost.color)
            self._draw_ghost_eyes(surface, cx, cy, r, ghost.dir)
            return

        if ghost.mode == GHOST_DEAD:
            cx = int(ghost.px * t + t // 2)
            cy = int(ghost.py * t + t // 2)
            r  = max(4, t // 2 - 2)
            self._draw_ghost_eyes(surface, cx, cy, r, ghost.dir)
            return

        cx = int(ghost.px * t + t // 2)
        cy = int(ghost.py * t + t // 2)
        r  = max(4, t // 2 - 2)

        if ghost.mode == GHOST_FRIGHTENED:
            flash = (
                ghost.fright_timer < GHOST_FRIGHT_FLASH_T
                and int(ghost.fright_timer * 5) % 2 == 0
            )
            color = GHOST_FLASH_COL if flash else GHOST_FRIGHT_COL
        else:
            color = ghost.color

        self._ghost_body(surface, cx, cy, r, color)

        if ghost.mode == GHOST_FRIGHTENED:
            # Two white eyes + squiggly mouth
            eye_y = cy - r // 4
            for ex in (cx - r // 3, cx + r // 3):
                pygame.draw.circle(surface, GHOST_EYE_WHITE, (ex, eye_y), max(2, r // 5))
            mouth_y = cy + r // 4
            for i in range(4):
                x0 = cx - r // 3 + i * (r // 6)
                y0 = mouth_y + (0 if i % 2 == 0 else r // 8)
                x1 = x0 + r // 6
                y1 = mouth_y + (r // 8 if i % 2 == 0 else 0)
                pygame.draw.line(surface, GHOST_EYE_WHITE, (x0, y0), (x1, y1), 1)
        else:
            self._draw_ghost_eyes(surface, cx, cy, r, ghost.dir)

    def _ghost_body(
        self,
        surface: "pygame.Surface",
        cx: int, cy: int, r: int,
        color: Tuple,
    ):
        """Classic ghost silhouette: rounded head + rectangular body + wavy skirt."""
        head_cy = cy - r // 4
        pygame.draw.circle(surface, color, (cx, head_cy), r)
        body_top = head_cy
        body_h   = r + r // 2
        pygame.draw.rect(surface, color, (cx - r, body_top, r * 2, body_h))
        skirt_y = body_top + body_h
        bump_r  = max(2, r // 3)
        for bx in (cx - r + bump_r, cx, cx + r - bump_r):
            pygame.draw.circle(surface, color, (bx, skirt_y), bump_r)

    def _draw_ghost_eyes(
        self,
        surface: "pygame.Surface",
        cx: int, cy: int, r: int,
        direction: Tuple[int, int],
    ):
        """Sclera + iris pointing in the ghost's movement direction."""
        eye_y    = cy - r // 2
        sclera_r = max(2, r // 3)
        iris_r   = max(1, r // 5)
        dx = direction[0] if direction != DIR_NONE else 0
        dy = direction[1] if direction != DIR_NONE else 0
        for ex in (cx - r // 3, cx + r // 3):
            pygame.draw.circle(surface, GHOST_EYE_WHITE, (ex, eye_y), sclera_r)
            pygame.draw.circle(
                surface, GHOST_EYE_IRIS,
                (ex + dx * iris_r, eye_y + dy * iris_r),
                iris_r,
            )

    # ------------------------------------------------------------------
    def _draw_hud(self, surface: "pygame.Surface", state: GameState):
        w = self.meta["width"]  * self.t
        h = self.meta["height"] * self.t
        font = pygame.font.SysFont("monospace", 18, bold=True)

        score_surf = font.render(f"SCORE  {state.score:>6}", True, WHITE)
        surface.blit(score_surf, (10, h + 8))

        # Lives
        lives_surf = font.render(f"LIVES  {state.lives}", True, YELLOW)
        surface.blit(lives_surf, (w // 2 - 50, h + 8))

        if state.won:
            win_surf = font.render("YOU WIN!   R = restart", True, YELLOW)
            surface.blit(win_surf, (w // 2 - 110, h + 8))

        if state.game_over:
            go_surf = font.render("GAME OVER   R = restart", True, (255, 0, 0))
            surface.blit(go_surf, (w // 2 - 120, h + 8))

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
