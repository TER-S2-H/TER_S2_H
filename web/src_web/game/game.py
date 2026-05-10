import sys
import os
import math
import asyncio
import pygame

from .environment import (
    Environment, GHOST_START_POSITIONS,
    CLOUD_API_URL, LOCAL_API_URL,
)
from .pacman_agent import PacManAgent
from .ghost import Ghost, GHOST_COLORS, get_ghost_agent
from .recorder import Recorder
from .agent import Direction, DIRECTION_VECTORS

# ── Timing ─────────────────────────────────────────────────────────────────
FPS             = 60
FRAMES_PER_TICK = 8

# ── Mise en page ────────────────────────────────────────────────────────────
TILE_SIZE     = 16
HEADER_HEIGHT = 50

# ── Rendu des blocs murs ─────────────────────────────────────────────────────
WALL_MARGIN = 1
WALL_BR     = 5
WALL_FILL   = (125, 165, 255,  0)
WALL_BORDER = ( 45,  75, 185, 220)

C = {
    "bg":           (  0,   0,   0),
    "pacman":       (255, 220,   0),
    "pellet":       (255, 185, 130),
    "power_pellet": (255, 255, 255),
    "frightened":   ( 30,  30, 200),
    "fright_blink": (200, 200, 255),
    "text":         (255, 255, 255),
    "text_yellow":  (255, 220,   0),
    "ghost_wall":   (255, 182, 193),
    "header_sep":   ( 33,  66, 255),
}

BONUS_LIFE_THRESHOLD = 10_000


class Game:
    """
    Version web (Pygbag) de la boucle principale.

    Différences par rapport à src/game/game.py :
      - run() est async : await asyncio.sleep(0) à chaque frame → contrôle cédé au navigateur
      - load_maze() est attendu avec await
      - pygame.font.SysFont → pygame.font.Font(None, size) (polices système absentes en WASM)
      - sys.exit() → return (sys.exit interrompt le runtime WebAssembly)
      - Recorder.save() déclenche un téléchargement navigateur au lieu d'écrire sur disque
    """

    def __init__(self, seed=None, local=False, record=False,
                 pacman_agent=None, no_ghosts=False, ai="random"):
        self.base_url       = LOCAL_API_URL if local else CLOUD_API_URL
        self.seed           = seed
        self.record         = record
        self._pacman_agent  = pacman_agent
        self._no_ghosts     = no_ghosts
        self._ai            = ai
        self._bonus_given   = False
        self._frame         = 0
        self._maze_surf     = None
        self._caught_coups  = None
        self._coups         = 0
        self._font_small    = None

    # ── Boucle principale (async pour Pygbag) ───────────────────────────────
    async def run(self):
        pygame.init()

        env = Environment(seed=self.seed, base_url=self.base_url)
        print("Chargement du labyrinthe…")
        try:
            await env.load_maze()
        except Exception as e:
            print(f"Erreur API : {e}")
            pygame.quit()
            return  # sys.exit(1) → return en WebAssembly

        print(f"seed={env.seed}  ({env.width}×{env.height})")

        pacman = self._pacman_agent or PacManAgent()
        ghosts = ([] if self._no_ghosts else
                  [get_ghost_agent(self._ai, i, gx, gy)
                   for i, (gx, gy) in enumerate(GHOST_START_POSITIONS)])
        env.ghosts = ghosts
        recorder = Recorder(seed=env.seed) if self.record else None

        win_w = env.width  * TILE_SIZE
        win_h = env.height * TILE_SIZE + HEADER_HEIGHT
        screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption(f"Pac-Man  —  seed {env.seed}")
        clock = pygame.time.Clock()

        # pygame.font.SysFont n'est pas disponible en WebAssembly → Font(None, …)
        font             = pygame.font.Font(None, 20)
        font_big         = pygame.font.Font(None, 48)
        self._font_small = pygame.font.Font(None, 16)

        paused  = False
        running = True

        try:
            while running:
                clock.tick(FPS)
                self._frame += 1

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            running = False
                        if event.key == pygame.K_p:
                            paused = not paused
                        if event.key == pygame.K_r and env.game_over:
                            env = Environment(seed=self.seed, base_url=self.base_url)
                            await env.load_maze()
                            ghosts = ([] if self._no_ghosts else
                                      [get_ghost_agent(self._ai, i, gx, gy)
                                       for i, (gx, gy) in enumerate(GHOST_START_POSITIONS)])
                            env.ghosts = ghosts
                            pacman = self._pacman_agent or PacManAgent()
                            self._bonus_given  = False
                            self._caught_coups = None
                            self._coups        = 0
                            self._frame = 0
                            self._maze_surf = None
                            if self.record:
                                recorder = Recorder(seed=env.seed)
                            pygame.display.set_caption(f"Pac-Man  —  seed {env.seed}")
                        pacman.handle_event(event)

                if env.game_over:
                    self._draw(screen, env, font)
                    msg      = font_big.render("GAME OVER", True, (255, 0, 0))
                    coups_txt = (f"Attrapé en {self._caught_coups} coups"
                                 if self._caught_coups is not None else "")
                    caught_s = font.render(coups_txt, True, C["text_yellow"])
                    hint     = font.render("R = rejouer  |  Q = quitter", True, C["text"])
                    y_msg    = win_h//2 - msg.get_height()//2
                    y_caught = y_msg + msg.get_height() + 8
                    y_hint   = y_caught + caught_s.get_height() + 6
                    screen.blit(msg,      (win_w//2 - msg.get_width()//2,      y_msg))
                    screen.blit(caught_s, (win_w//2 - caught_s.get_width()//2, y_caught))
                    screen.blit(hint,     (win_w//2 - hint.get_width()//2,     y_hint))
                    pygame.display.flip()
                    await asyncio.sleep(0)
                    continue

                if paused:
                    self._draw(screen, env, font)
                    msg = font_big.render("PAUSE", True, C["text_yellow"])
                    screen.blit(msg, (win_w//2 - msg.get_width()//2,
                                      win_h//2 - msg.get_height()//2))
                    pygame.display.flip()
                    await asyncio.sleep(0)
                    continue

                if env.level_complete:
                    env.next_level()
                    self._bonus_given = False
                    self._maze_surf = None
                    await asyncio.sleep(0)
                    continue

                if self._frame % FRAMES_PER_TICK == 0:
                    state = env.get_state()
                    actions = {"pacman": pacman.get_action(state)}
                    for i, g in enumerate(ghosts):
                        actions[f"ghost_{i}"] = g.get_action(state)

                    px_before, py_before = env.pacman_x, env.pacman_y
                    env.update(actions)
                    if (env.pacman_x, env.pacman_y) != (px_before, py_before):
                        self._coups += 1
                    if env.game_over and self._caught_coups is None:
                        self._caught_coups = self._coups
                    pacman.confirm_direction(env.pacman_dir)

                    if not self._bonus_given and env.score >= BONUS_LIFE_THRESHOLD:
                        env.lives += 1
                        self._bonus_given = True
                        print("[Game] Vie bonus accordée !")

                    if recorder is not None:
                        recorder.record(actions, env.get_grid_snapshot())

                self._draw(screen, env, font)
                pygame.display.flip()

                # CRITIQUE pour Pygbag : cède le contrôle au navigateur à chaque frame
                await asyncio.sleep(0)

        except KeyboardInterrupt:
            pass
        finally:
            if recorder is not None:
                recorder.coups = (self._caught_coups
                                  if self._caught_coups is not None
                                  else self._coups)
                save_path = os.path.abspath("recording.json")
                print(f"Sauvegarde en cours... → {save_path}")
                recorder.save(save_path)
            pygame.quit()

    # ── Rendu principal (identique à src/game/game.py) ──────────────────────
    def _draw(self, screen, env, font):
        screen.fill(C["bg"])
        oy = HEADER_HEIGHT

        maze_surf = self._get_maze_surf(env)
        screen.blit(maze_surf, (0, oy))

        self._draw_ghost_door(screen, env, oy)
        self._draw_pellets(screen, env, oy)

        for i, ghost in enumerate(env.ghosts):
            self._draw_ghost(screen, ghost, i, env.frightened_timer, oy)

        self._draw_pacman(screen, env, oy)
        self._draw_header(screen, env, font)

    def _get_maze_surf(self, env):
        if self._maze_surf is not None:
            return self._maze_surf

        TS = TILE_SIZE
        M  = WALL_MARGIN
        BR = WALL_BR

        w = env.width  * TS
        h = env.height * TS

        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))

        for (wx, wy) in env.walls:
            rect = pygame.Rect(wx*TS + M, wy*TS + M, TS - 2*M, TS - 2*M)
            pygame.draw.rect(surf, WALL_FILL,   rect, border_radius=BR)
            pygame.draw.rect(surf, WALL_BORDER, rect, width=3, border_radius=BR)

        self._maze_surf = surf
        return surf

    def _draw_ghost_door(self, screen, env, oy):
        TS = TILE_SIZE
        for (x, y) in env.ghost_walls:
            mid = y*TS + oy + TS//2 - 1
            pygame.draw.line(screen, C["ghost_wall"],
                             (x*TS, mid), (x*TS + TS, mid), 3)

    def _draw_pellets(self, screen, env, oy):
        TS = TILE_SIZE
        power_visible = (self._frame % 30) < 22

        for (px, py), pellet in env.pellets.items():
            cx = px*TS + TS//2
            cy = py*TS + TS//2 + oy
            if pellet.type == "power":
                if power_visible:
                    pygame.draw.circle(screen, C["power_pellet"], (cx, cy), TS//4)
            else:
                pygame.draw.circle(screen, C["pellet"], (cx, cy), 2)

    def _draw_pacman(self, screen, env, oy):
        TS  = TILE_SIZE
        r   = TS//2 - 1
        cx  = env.pacman_x * TS + TS//2
        cy  = env.pacman_y * TS + TS//2 + oy

        mouth_deg = abs(math.sin(self._frame * 0.18)) * 38 + 4
        mouth_rad = math.radians(mouth_deg)

        dir_angle = {
            Direction.RIGHT:  0,
            Direction.LEFT:   math.pi,
            Direction.UP:     math.pi / 2,
            Direction.DOWN:  -math.pi / 2,
            Direction.NONE:   0,
        }.get(env.pacman_dir, 0)

        a_start  = dir_angle + mouth_rad
        arc_span = 2*math.pi - 2*mouth_rad
        n = 24

        pts = [(cx, cy)]
        for i in range(n + 1):
            a = a_start + arc_span * i / n
            pts.append((cx + r*math.cos(a),
                        cy - r*math.sin(a)))

        pygame.draw.polygon(screen, C["pacman"], pts)

    def _draw_ghost(self, screen, ghost, idx, frightened_timer, oy):
        TS = TILE_SIZE
        r  = TS//2 - 1
        cx = ghost.x * TS + TS//2
        cy = ghost.y * TS + TS//2 + oy

        if ghost.frightened:
            if frightened_timer < 20 and (self._frame // 6) % 2:
                body_color = C["fright_blink"]
            else:
                body_color = C["frightened"]
        else:
            body_color = GHOST_COLORS[idx % len(GHOST_COLORS)]

        pygame.draw.circle(screen, body_color, (cx, cy), r)
        pygame.draw.rect(screen, body_color, (cx-r, cy, r*2, r))

        seg = (r * 2 + 2) // 3
        for i in range(3):
            bx = cx - r + i*seg + seg//2
            pygame.draw.circle(screen, C["bg"], (bx, cy + r), seg//2 + 1)

        ew = max(r // 3, 2)
        ep = max(ew // 2, 1)

        for ex in [cx - r//3, cx + r//3]:
            if ghost.frightened:
                pygame.draw.circle(screen, (200, 30, 30), (ex, cy - r//5), ep + 1)
            else:
                ey = cy - r // 3
                pygame.draw.circle(screen, (255, 255, 255), (ex, ey), ew)
                dv = DIRECTION_VECTORS.get(ghost._last_direction, (0, 0))
                dx, dy = dv if dv != (0, 0) else (1, 0)
                px = ex + dx * (ew - ep)
                py = ey + dy * (ew - ep)
                pygame.draw.circle(screen, (0, 0, 210), (px, py), ep)

    def _draw_header(self, screen, env, font):
        pygame.draw.rect(screen, C["bg"],
                         (0, 0, screen.get_width(), HEADER_HEIGHT))
        pygame.draw.line(screen, C["header_sep"],
                         (0, HEADER_HEIGHT - 1),
                         (screen.get_width(), HEADER_HEIGHT - 1), 1)

        score_s = font.render(f"SCORE  {env.score:>6}", True, C["text_yellow"])
        screen.blit(score_s, (10, HEADER_HEIGHT//2 - score_s.get_height()//2))

        level_s = font.render(f"NIVEAU {env.level}", True, C["text"])
        screen.blit(level_s, (screen.get_width()//2 - level_s.get_width()//2,
                               HEADER_HEIGHT//2 - level_s.get_height()//2))

        r_life = 7
        for i in range(env.lives):
            lx = screen.get_width() - 15 - i * (r_life*2 + 4)
            ly = HEADER_HEIGHT // 2
            pts = [(lx, ly)]
            for j in range(25):
                a = math.radians(35 + 290 * j / 24)
                pts.append((lx + r_life*math.cos(a),
                             ly - r_life*math.sin(a)))
            pygame.draw.polygon(screen, C["pacman"], pts)

        if env.frightened_timer > 0:
            ft = font.render(f"*{env.frightened_timer:>3}", True, C["fright_blink"])
            screen.blit(ft, (screen.get_width()//2 + 80,
                              HEADER_HEIGHT//2 - ft.get_height()//2))

        if self._font_small is not None:
            coups = self._caught_coups if self._caught_coups is not None else self._coups
            coups_s = self._font_small.render(f"{coups} coups", True, (140, 140, 140))
            screen.blit(coups_s, (screen.get_width()//2 - coups_s.get_width()//2,
                                  HEADER_HEIGHT - coups_s.get_height() - 2))
