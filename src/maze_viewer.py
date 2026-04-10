"""
Affiche un labyrinthe Pac-Man dans une fenêtre pygame, sans Pac-Man ni fantômes.

Usage :
    python -m src.maze_viewer              # seed aléatoire
    python -m src.maze_viewer --seed 42    # seed fixe
    python -m src.maze_viewer --local      # API locale (localhost:5000)
"""

import sys
import math
import argparse
import pygame

from .game.environment import Environment, CLOUD_API_URL, LOCAL_API_URL

# ── Mise en page ─────────────────────────────────────────────────────────────
TILE_SIZE     = 16
HEADER_HEIGHT = 40

# ── Couleurs ──────────────────────────────────────────────────────────────────
BG            = (  0,   0,   0)
WALL_FILL     = (125, 165, 255,   0)
WALL_BORDER   = ( 45,  75, 185, 220)
PELLET_COLOR  = (255, 185, 130)
POWER_COLOR   = (255, 255, 255)
GHOST_DOOR    = (255, 182, 193)
TEXT_COLOR    = (255, 255, 255)
TEXT_YELLOW   = (255, 220,   0)

WALL_MARGIN = 1
WALL_BR     = 5


def _build_maze_surf(env):
    TS, M, BR = TILE_SIZE, WALL_MARGIN, WALL_BR
    surf = pygame.Surface((env.width * TS, env.height * TS), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    for (wx, wy) in env.walls:
        rect = pygame.Rect(wx*TS + M, wy*TS + M, TS - 2*M, TS - 2*M)
        pygame.draw.rect(surf, WALL_FILL,   rect, border_radius=BR)
        pygame.draw.rect(surf, WALL_BORDER, rect, width=3, border_radius=BR)
    return surf


def _draw_ghost_door(screen, env, oy):
    TS = TILE_SIZE
    for (x, y) in env.ghost_walls:
        mid = y*TS + oy + TS//2 - 1
        pygame.draw.line(screen, GHOST_DOOR,
                         (x*TS, mid), (x*TS + TS, mid), 3)


def _draw_pellets(screen, env, oy, frame):
    TS = TILE_SIZE
    power_visible = (frame % 30) < 22
    for (px, py), pellet in env.pellets.items():
        cx = px*TS + TS//2
        cy = py*TS + TS//2 + oy
        if pellet.type == "power":
            if power_visible:
                pygame.draw.circle(screen, POWER_COLOR, (cx, cy), TS//4)
        else:
            pygame.draw.circle(screen, PELLET_COLOR, (cx, cy), 2)


def _draw_header(screen, env, font):
    pygame.draw.rect(screen, BG, (0, 0, screen.get_width(), HEADER_HEIGHT))
    pygame.draw.line(screen, (33, 66, 255),
                     (0, HEADER_HEIGHT - 1),
                     (screen.get_width(), HEADER_HEIGHT - 1), 1)
    label = font.render(
        f"Pac-Man Maze  —  seed {env.seed}   ({env.width}×{env.height})",
        True, TEXT_YELLOW,
    )
    screen.blit(label, (10, HEADER_HEIGHT//2 - label.get_height()//2))


def run(seed=None, local=False):
    base_url = LOCAL_API_URL if local else CLOUD_API_URL

    pygame.init()
    font = pygame.font.SysFont("monospace", 13, bold=True)

    env = Environment(seed=seed, base_url=base_url)
    print("Chargement du labyrinthe…")
    try:
        env.load_maze()
    except Exception as e:
        print(f"Erreur API : {e}")
        pygame.quit()
        sys.exit(1)
    print(f"seed={env.seed}  ({env.width}×{env.height})")

    win_w = env.width  * TILE_SIZE
    win_h = env.height * TILE_SIZE + HEADER_HEIGHT
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"Maze viewer  —  seed {env.seed}")
    clock = pygame.time.Clock()

    maze_surf = _build_maze_surf(env)
    frame = 0

    while True:
        clock.tick(30)
        frame += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    sys.exit(0)

        oy = HEADER_HEIGHT
        screen.fill(BG)
        screen.blit(maze_surf, (0, oy))
        _draw_ghost_door(screen, env, oy)
        _draw_pellets(screen, env, oy, frame)
        _draw_header(screen, env, font)
        pygame.display.flip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pac-Man maze viewer (sans personnages)")
    parser.add_argument("--seed",  type=int, default=None, help="Seed du labyrinthe")
    parser.add_argument("--local", action="store_true",    help="Utiliser l'API locale")
    args = parser.parse_args()
    run(seed=args.seed, local=args.local)
