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

# -------------------------------------------------------------------
# Configuration des URLs
# -------------------------------------------------------------------
CLOUD_URL = "https://pacman-s2-h.onrender.com"
LOCAL_URL = "http://localhost:5000"

# -------------------------------------------------------------------
# Couleurs Pac-Man
# -------------------------------------------------------------------
COLOR_BACKGROUND = (0, 0, 0)        # Noir
COLOR_WALL = (33, 33, 222)          # Bleu Pac-Man
COLOR_PATH = (0, 0, 0)              # Noir (couloir vide)
COLOR_GHOST_WALL = (255, 182, 193)  # Rose (porte fantôme)
COLOR_INFO_TEXT = (255, 255, 255)   # Blanc
COLOR_SEED_TEXT = (255, 255, 0)     # Jaune


def fetch_maze(base_url: str, seed: int = None) -> dict:
    """
    Appelle l'API pour récupérer un maze en JSON.
    
    Args:
        base_url : URL de base de l'API (cloud ou local)
        seed     : graine optionnelle pour la génération
        
    Returns:
        Le maze sous forme de dictionnaire Python
    """
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
        print(f"Erreur : Le serveur {base_url} met trop de temps à répondre (>30s)")
        print("  → Sur Render (plan gratuit), le service se met en veille. Réessayez dans 30s.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"Erreur HTTP : {e}")
        sys.exit(1)


def render_maze(screen, maze_data: dict, tile_size: int):
    """
    Dessine le maze dans la fenêtre Pygame.
    
    On reçoit le JSON de l'API qui contient :
      - walls      : liste de cases {x, y} qui sont des murs
      - ghost_walls: liste de cases {x, y} qui sont des portes fantômes
      - metadata   : width, height, node_count, edge_count
    
    Tout ce qui n'est pas un mur est un couloir (fond noir).
    """
    screen.fill(COLOR_BACKGROUND)

    # Dessiner les murs
    for wall in maze_data.get("walls", []):
        x = wall["x"] * tile_size
        y = wall["y"] * tile_size
        pygame.draw.rect(screen, COLOR_WALL, (x, y, tile_size, tile_size))

    # Dessiner la porte de la maison des fantômes
    for gw in maze_data.get("ghost_walls", []):
        x = gw["x"] * tile_size
        y = gw["y"] * tile_size + tile_size // 2 - 1
        pygame.draw.rect(screen, COLOR_GHOST_WALL, (x, y, tile_size, 3))


def draw_info(screen, maze_data: dict, base_url: str, font):
    """
    Affiche les informations du maze en bas de l'écran.
    """
    meta = maze_data.get("metadata", {})
    seed = maze_data.get("seed", "?")
    width = meta.get("width", "?")
    height = meta.get("height", "?")
    nodes = meta.get("node_count", "?")

    info = f"Taille: {width}x{height}  |  Noeuds: {nodes}  |  Source: {base_url}"
    seed_text = f"Seed: {seed}  (appuyez sur ESPACE pour un nouveau maze, Q pour quitter)"

    screen_w = screen.get_width()
    screen_h = screen.get_height()

    # Fond semi-transparent en bas
    bar_height = 45
    bar_surf = pygame.Surface((screen_w, bar_height), pygame.SRCALPHA)
    bar_surf.fill((0, 0, 0, 180))
    screen.blit(bar_surf, (0, screen_h - bar_height))

    text1 = font.render(info, True, COLOR_INFO_TEXT)
    text2 = font.render(seed_text, True, COLOR_SEED_TEXT)
    screen.blit(text1, (10, screen_h - bar_height + 5))
    screen.blit(text2, (10, screen_h - bar_height + 25))


def main():
    # --- Arguments CLI ---
    parser = argparse.ArgumentParser(
        description="Client Pygame — Visualisation du maze Pac-Man via API"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Utiliser le serveur local (localhost:5000) au lieu du cloud"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Graine pour la génération du maze (optionnel)"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=20,
        help="Taille d'une case en pixels (défaut: 20)"
    )
    args = parser.parse_args()

    base_url = LOCAL_URL if args.local else CLOUD_URL
    tile_size = args.tile_size
    current_seed = args.seed

    # --- Récupération du premier maze ---
    maze_data = fetch_maze(base_url, current_seed)
    meta = maze_data.get("metadata", {})
    width = meta.get("width", 28)
    height = meta.get("height", 31)

    # --- Initialisation Pygame ---
    pygame.init()
    font = pygame.font.SysFont("monospace", 13)

    win_w = width * tile_size
    win_h = height * tile_size + 50   # +50 pour la barre d'info en bas
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"Pac-Man Maze — seed {maze_data.get('seed', '?')}")
    clock = pygame.time.Clock()

    # --- Boucle principale ---
    running = True
    while running:
        for event in pygame.event.get():

            # Quitter avec la croix ou Q
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

                # ESPACE → nouveau maze aléatoire
                if event.key == pygame.K_SPACE:
                    print("Génération d'un nouveau maze...")
                    maze_data = fetch_maze(base_url, seed=None)
                    meta = maze_data.get("metadata", {})
                    pygame.display.set_caption(
                        f"Pac-Man Maze — seed {maze_data.get('seed', '?')}"
                    )

        # Dessiner le maze et les infos
        render_maze(screen, maze_data, tile_size)
        draw_info(screen, maze_data, base_url, font)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
