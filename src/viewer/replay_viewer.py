"""
Pygame-based session replay viewer.

Controls:
  SPACE / P    — play / pause
  LEFT / A     — step backward
  RIGHT / D    — step forward
  HOME         — jump to frame 0
  END          — jump to last frame
  UP / DOWN    — speed +/-
  H            — toggle path heatmap overlay
  Q / Escape   — quit

Colour-coded ghosts: red, pink, cyan, orange (in order).
Frightened ghosts shown in blue; flashing when < 10 ticks remain.
"""

from __future__ import annotations

import math
import pickle
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import pygame
except ImportError:
    sys.exit("pygame is required: pip install pygame")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BLACK          = (  0,   0,   0)
DARK_BLUE      = ( 25,  25, 100)
WHITE          = (255, 255, 255)
YELLOW         = (255, 220,   0)
PELLET_COL     = (255, 255, 184)
ENERGY_COL     = (255, 255, 255)
GHOST_WALL_COL = (255, 182, 193)
HEATMAP_COL    = ( 60, 180,  60)

GHOST_COLORS   = [
    (255,  30,  30),   # red   — blinky
    (255, 184, 255),   # pink  — pinky
    (  0, 255, 255),   # cyan  — inky
    (255, 184,  82),   # orange— clyde
]
GHOST_FRIGHT_COL  = (  0,  50, 200)
GHOST_FLASH_COL   = (255, 255, 255)
GHOST_EYE_WHITE   = (255, 255, 255)
GHOST_EYE_IRIS    = (  0,   0, 200)

TILE = 20


class ReplayViewer:
    """Load a session pickle and display it frame by frame."""

    def __init__(self, session: Dict[str, Any], tile_size: int = TILE):
        self.session = session
        self.frames  = session["frames"]
        self.t       = tile_size

        # Derive maze dimensions from graph metadata
        coords = []
        if "index_to_coord" in session.get("graph", {}):
            coords = [tuple(v) for v in session["graph"]["index_to_coord"].values()]
        if coords:
            self.width  = max(c[0] for c in coords) + 2
            self.height = max(c[1] for c in coords) + 2
        else:
            self.width  = 28
            self.height = 31

        self._ghost_wall_nodes: Set[Tuple[int, int]] = {
            tuple(g) for g in session["graph"].get("ghost_wall_nodes", [])  # type: ignore
        }
        self._wall_cells = self._infer_walls(coords)

        # Initial pellets / energizers (for heatmap)
        self._init_pellets:    Set[Tuple] = {tuple(p) for p in session.get("initial_pellets",    [])}
        self._init_energizers: Set[Tuple] = {tuple(e) for e in session.get("initial_energizers", [])}

        self._frame_idx   = 0
        self._playing     = False
        self._speed       = 8    # frames per second (playback)
        self._show_heatmap = False

        # Build path heatmap (cells visited by Pac-Man at least once)
        self._heatmap: Set[Tuple] = {
            tuple(f["pacman_pos"]) for f in self.frames
        }

    # ------------------------------------------------------------------

    def _infer_walls(self, coords: List) -> Set[Tuple[int, int]]:
        """Infer wall cells: all grid cells not in the node set."""
        if not coords:
            return set()
        node_set = set(map(tuple, coords))
        walls: Set[Tuple[int, int]] = set()
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in node_set:
                    walls.add((x, y))
        return walls

    # ------------------------------------------------------------------

    def run(self):
        pygame.init()
        pygame.display.set_caption("Pac-Man Replay Viewer")
        t = self.t
        screen = pygame.display.set_mode(
            (self.width * t, self.height * t + 60)
        )
        clock  = pygame.time.Clock()
        font   = pygame.font.SysFont("monospace", 15, bold=True)
        big_font = pygame.font.SysFont("monospace", 18, bold=True)

        accum = 0.0

        while True:
            dt = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)

            if self._playing:
                accum += dt
                step_dt = 1.0 / max(1, self._speed)
                while accum >= step_dt:
                    accum -= step_dt
                    self._frame_idx = min(self._frame_idx + 1, len(self.frames) - 1)
                    if self._frame_idx == len(self.frames) - 1:
                        self._playing = False

            frame = self.frames[self._frame_idx]
            self._draw(screen, frame, font, big_font)
            pygame.display.flip()

    def _handle_key(self, key: int):
        n = len(self.frames)
        if key in (pygame.K_SPACE, pygame.K_p):
            self._playing = not self._playing
        elif key in (pygame.K_RIGHT, pygame.K_d):
            self._frame_idx = min(self._frame_idx + 1, n - 1)
        elif key in (pygame.K_LEFT, pygame.K_a):
            self._frame_idx = max(self._frame_idx - 1, 0)
        elif key == pygame.K_HOME:
            self._frame_idx = 0
        elif key == pygame.K_END:
            self._frame_idx = n - 1
        elif key in (pygame.K_UP,):
            self._speed = min(self._speed + 2, 60)
        elif key in (pygame.K_DOWN,):
            self._speed = max(self._speed - 2, 1)
        elif key == pygame.K_h:
            self._show_heatmap = not self._show_heatmap
        elif key in (pygame.K_q, pygame.K_ESCAPE):
            pygame.quit()
            raise SystemExit(0)

    # ------------------------------------------------------------------

    def _draw(self, surface, frame: dict, font, big_font):
        t = self.t
        surface.fill(BLACK)

        # Walls
        for (wx, wy) in self._wall_cells:
            if (wx, wy) not in self._ghost_wall_nodes:
                pygame.draw.rect(surface, DARK_BLUE,
                                 (wx * t, wy * t, t, t))

        # Ghost-house door
        for (gx, gy) in self._ghost_wall_nodes:
            mid_y = gy * t + t // 2
            pygame.draw.rect(surface, GHOST_WALL_COL,
                             (gx * t, mid_y - 2, t, 4))

        # Heatmap
        if self._show_heatmap:
            for pos in self._heatmap:
                px, py = pos[0] * t + t // 4, pos[1] * t + t // 4
                pygame.draw.rect(surface, HEATMAP_COL, (px, py, t // 2, t // 2))

        # Pellets
        pr = max(1, t // 10)
        for pellet in frame["active_pellets"]:
            cx, cy = pellet[0] * t + t // 2, pellet[1] * t + t // 2
            pygame.draw.circle(surface, PELLET_COL, (cx, cy), pr)

        # Energizers (pulsing)
        er = max(3, t // 5) + int(abs(math.sin(pygame.time.get_ticks() / 300.0)) * 3)
        for eng in frame["active_energizers"]:
            cx, cy = eng[0] * t + t // 2, eng[1] * t + t // 2
            pygame.draw.circle(surface, ENERGY_COL, (cx, cy), er)

        # Ghosts
        for i, gpos in enumerate(frame["ghost_positions"]):
            fright = frame["ghost_frightened"][i]
            self._draw_ghost(surface, gpos, fright, i)

        # Pac-Man
        self._draw_pacman(surface, tuple(frame["pacman_pos"]),
                          self._frame_idx, font)

        # HUD
        hud_y = self.height * t + 5
        step_s   = big_font.render(
            f"Step {frame['step']:>4} / {len(self.frames)-1}", True, WHITE)
        surface.blit(step_s, (5, hud_y))

        outcome  = self.session.get("outcome", "?")
        oc_col   = (255, 60, 60) if outcome == "eaten" else (60, 255, 60) if outcome == "cleared" else WHITE
        oc_s     = font.render(f"Outcome: {outcome}", True, oc_col)
        surface.blit(oc_s, (self.width * t - 180, hud_y))

        pellets_left = len(frame["active_pellets"]) + len(frame["active_energizers"])
        pl_s = font.render(f"Pellets left: {pellets_left}", True, PELLET_COL)
        surface.blit(pl_s, (5, hud_y + 20))

        speed_s = font.render(f"Speed: {self._speed} fps  H=heatmap", True, (180, 180, 180))
        surface.blit(speed_s, (5, hud_y + 38))

        play_s = font.render("PLAY" if self._playing else "PAUSED", True,
                             (100, 255, 100) if self._playing else (255, 200, 60))
        surface.blit(play_s, (self.width * t - 80, hud_y + 20))

    def _draw_pacman(self, surface, pos: tuple, frame_idx: int, font):
        t  = self.t
        cx = pos[0] * t + t // 2
        cy = pos[1] * t + t // 2
        r  = max(4, t // 2 - 2)

        phase      = (frame_idx % 20) / 20.0 * 2 * math.pi
        mouth_deg  = int(abs(math.sin(phase)) * 35)
        start_a    = math.radians(mouth_deg)
        end_a      = math.radians(360 - mouth_deg)

        points = [(cx, cy)]
        segs   = 36
        for i in range(segs + 1):
            a = start_a + (end_a - start_a) * i / segs
            points.append((cx + int(r * math.cos(a)), cy - int(r * math.sin(a))))
        if len(points) > 2:
            pygame.draw.polygon(surface, YELLOW, points)

    def _draw_ghost(self, surface, pos: tuple, fright: int, idx: int):
        t  = self.t
        cx = pos[0] * t + t // 2
        cy = pos[1] * t + t // 2
        r  = max(4, t // 2 - 2)

        if fright > 0:
            flash = fright < 10 and int(pygame.time.get_ticks() / 200) % 2 == 0
            color = GHOST_FLASH_COL if flash else GHOST_FRIGHT_COL
        else:
            color = GHOST_COLORS[idx % len(GHOST_COLORS)]

        # Body
        head_cy = cy - r // 4
        pygame.draw.circle(surface, color, (cx, head_cy), r)
        pygame.draw.rect(surface, color, (cx - r, head_cy, r * 2, r + r // 2))
        skirt_y = head_cy + r + r // 2
        bump_r  = max(2, r // 3)
        for bx in (cx - r + bump_r, cx, cx + r - bump_r):
            pygame.draw.circle(surface, color, (bx, skirt_y), bump_r)

        # Eyes (only when not fully frightened)
        if fright == 0:
            eye_y    = head_cy - r // 4
            sclera_r = max(2, r // 3)
            iris_r   = max(1, r // 5)
            for ex in (cx - r // 3, cx + r // 3):
                pygame.draw.circle(surface, GHOST_EYE_WHITE, (ex, eye_y), sclera_r)
                pygame.draw.circle(surface, GHOST_EYE_IRIS,  (ex, eye_y), iris_r)


# ---------------------------------------------------------------------------
# Entry point (called from scripts/replay.py)
# ---------------------------------------------------------------------------

def load_and_run(session_path: str, tile_size: int = TILE):
    with open(session_path, "rb") as f:
        session = pickle.load(f)
    viewer = ReplayViewer(session, tile_size=tile_size)
    viewer.run()
