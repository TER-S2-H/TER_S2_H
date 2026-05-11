#!/usr/bin/env python3
"""
Client local Pygame — Visualisation du maze Pac-Man
Récupère le maze via l'API HTTP et l'affiche dans une fenêtre interactive.

Usage :
  python client.py                          # maze aléatoire depuis le cloud
  python client.py --seed 42               # seed fixe depuis le cloud
  python client.py --local                 # utilise le serveur local (localhost:5000)
  python client.py --local --seed 42       # seed fixe en local
"""

import argparse
import sys
import requests
import pygame

CLOUD_URL = "https://pacman-s2-h.onrender.com"
LOCAL_URL = "http://localhost:5000"

TILE_SIZE     = 16
HEADER_HEIGHT = 50
FOOTER_HEIGHT = 30

WALL_MARGIN = 1
WALL_BR     = 5
WALL_FILL   = (125, 165, 255,   0)
WALL_BORDER = ( 45,  75, 185, 220)

# Positions fixes des power pellets dans le maze Pac-Man classique
POWER_PELLET_POSITIONS = {(1, 3), (26, 3), (1, 23), (26, 23)}

# Touches de notation : K_0..K_5 → valeur entière
RATING_KEYS = {
    pygame.K_0: 0, pygame.K_1: 1, pygame.K_2: 2,
    pygame.K_3: 3, pygame.K_4: 4, pygame.K_5: 5,
}

C = {
    "bg":           (  0,   0,   0),
    "ghost_wall":   (255, 182, 193),
    "pellet":       (255, 185, 130),
    "power_pellet": (255, 255, 255),
    "text":         (255, 255, 255),
    "text_yellow":  (255, 220,   0),
    "header_sep":   ( 33,  66, 255),
    "footer_bg":    ( 10,  10,  30),
}


def fetch_maze(base_url: str, seed: int = None) -> dict:
    url = f"{base_url}/maze"
    params = {}
    if seed is not None:
        params["seed"] = seed

    print(f"Appel API : {url} {params if params else '(seed aléatoire)'}")

    try:
        response = requests.get(url, params=params, timeout=50)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print(f"Erreur : Impossible de se connecter à {base_url}")
        print("  → Si vous utilisez --local, lancez d'abord : python app.py")
        print("  → Si vous utilisez le cloud, vérifiez votre connexion internet")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"Erreur : Le serveur {base_url} met trop de temps à répondre (>50s)")
        print("  → Sur Render (plan gratuit), le service se met en veille. Réessayez dans 30s.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"Erreur HTTP : {e}")
        sys.exit(1)


def rate_maze(base_url: str, seed: int, rating: int) -> bool:
    """Envoie la note via PATCH /maze/<seed>/rate. Retourne True si succès."""
    try:
        resp = requests.patch(
            f"{base_url}/maze/{seed}/rate",
            json={"rating": rating},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la notation : {e}")
        return False


def build_maze_surf(maze_data: dict, tile_size: int) -> pygame.Surface:
    meta   = maze_data.get("metadata", {})
    width  = meta.get("width",  28)
    height = meta.get("height", 31)
    walls  = [(w["x"], w["y"]) for w in maze_data.get("walls", [])]

    TS = tile_size
    M  = WALL_MARGIN
    BR = WALL_BR

    surf = pygame.Surface((width * TS, height * TS), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))

    for (wx, wy) in walls:
        rect = pygame.Rect(wx*TS + M, wy*TS + M, TS - 2*M, TS - 2*M)
        pygame.draw.rect(surf, WALL_FILL,   rect, border_radius=BR)
        pygame.draw.rect(surf, WALL_BORDER, rect, width=3, border_radius=BR)

    return surf


def draw_pellets(screen, maze_data: dict, tile_size: int, oy: int):
    TS = tile_size
    for node in maze_data.get("nodes", []):
        if node.get("type") != "PATH":
            continue   # PATHBLANK = maison des fantômes, pas de pellets
        x, y = node["x"], node["y"]
        cx = x * TS + TS // 2
        cy = y * TS + TS // 2 + oy
        if (x, y) in POWER_PELLET_POSITIONS:
            pygame.draw.circle(screen, C["power_pellet"], (cx, cy), 7)
        else:
            pygame.draw.circle(screen, C["pellet"], (cx, cy), 3)


def draw_ghost_door(screen, maze_data: dict, tile_size: int, oy: int):
    TS = tile_size
    for gw in maze_data.get("ghost_walls", []):
        x, y = gw["x"], gw["y"]
        mid = y*TS + oy + TS//2 - 1
        pygame.draw.line(screen, C["ghost_wall"],
                         (x*TS, mid), (x*TS + TS, mid), 3)


def draw_footer(screen, font, rating_state: dict):
    W = screen.get_width()
    H = screen.get_height()
    y = H - FOOTER_HEIGHT

    pygame.draw.rect(screen, C["footer_bg"], (0, y, W, FOOTER_HEIGHT))
    pygame.draw.line(screen, C["header_sep"], (0, y), (W, y), 1)

    if rating_state["confirmed"]:
        text  = f"Note envoyée : {rating_state['value']}/5  ✓"
        color = C["text_yellow"]
    else:
        text  = "Notez ce maze : appuyez sur 0 à 5"
        color = C["text"]

    msg = font.render(text, True, color)
    screen.blit(msg, (W // 2 - msg.get_width() // 2,
                      y + FOOTER_HEIGHT // 2 - msg.get_height() // 2))


def draw_header(screen, maze_data: dict, base_url: str, font):
    seed = maze_data.get("seed", "?")
    W = screen.get_width()

    pygame.draw.rect(screen, C["bg"], (0, 0, W, HEADER_HEIGHT))
    pygame.draw.line(screen, C["header_sep"],
                     (0, HEADER_HEIGHT - 1), (W, HEADER_HEIGHT - 1), 1)

    # Ligne 1 : seed à gauche, source à droite
    seed_s = font.render(f"SEED  {seed}", True, C["text_yellow"])
    src    = base_url.replace("https://", "").replace("http://", "")
    src_s  = font.render(src, True, C["text"])
    screen.blit(seed_s, (10, 8))
    screen.blit(src_s,  (W - src_s.get_width() - 10, 8))

    # Ligne 2 : contrôles centrés
    hint_s = font.render("ESPACE = nouveau maze  |  Q = quitter", True, C["text"])
    screen.blit(hint_s, (W//2 - hint_s.get_width()//2, 30))


def main():
    parser = argparse.ArgumentParser(
        description="Client Pygame — Visualisation du maze Pac-Man via API"
    )
    parser.add_argument("--local",     action="store_true",
                        help="Utiliser le serveur local (localhost:5000) au lieu du cloud")
    parser.add_argument("--seed",      type=int, default=None,
                        help="Graine pour la génération du maze (optionnel)")
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE,
                        help=f"Taille d'une case en pixels (défaut: {TILE_SIZE})")
    args = parser.parse_args()

    base_url     = LOCAL_URL if args.local else CLOUD_URL
    tile_size    = args.tile_size
    current_seed = args.seed

    maze_data = fetch_maze(base_url, current_seed)
    meta   = maze_data.get("metadata", {})
    width  = meta.get("width",  28)
    height = meta.get("height", 31)

    pygame.init()
    font = pygame.font.SysFont("monospace", 14, bold=True)

    win_w = width  * tile_size
    win_h = height * tile_size + HEADER_HEIGHT + FOOTER_HEIGHT
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"Pac-Man Maze  —  seed {maze_data.get('seed', '?')}")
    clock = pygame.time.Clock()

    maze_surf   = build_maze_surf(maze_data, tile_size)
    oy          = HEADER_HEIGHT
    rating_state = {"confirmed": False, "value": None}

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                if event.key == pygame.K_SPACE:
                    print("Génération d'un nouveau maze...")
                    maze_data    = fetch_maze(base_url, seed=None)
                    maze_surf    = build_maze_surf(maze_data, tile_size)
                    rating_state = {"confirmed": False, "value": None}
                    pygame.display.set_caption(
                        f"Pac-Man Maze  —  seed {maze_data.get('seed', '?')}"
                    )
                if event.key in RATING_KEYS and not rating_state["confirmed"]:
                    rating = RATING_KEYS[event.key]
                    seed   = maze_data.get("seed")
                    print(f"Envoi de la note {rating}/5 pour le seed {seed}…")
                    if rate_maze(base_url, seed, rating):
                        rating_state = {"confirmed": True, "value": rating}
                    else:
                        print("Échec de l'envoi de la note.")

        screen.fill(C["bg"])
        screen.blit(maze_surf, (0, oy))
        draw_ghost_door(screen, maze_data, tile_size, oy)
        draw_pellets(screen, maze_data, tile_size, oy)
        draw_header(screen, maze_data, base_url, font)
        draw_footer(screen, font, rating_state)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
