import sys
import pygame

from .environment import (
    Environment, GHOST_START_POSITIONS,
    CLOUD_API_URL, LOCAL_API_URL,
)
from .pacman_agent import PacManAgent
from .ghost import Ghost, GHOST_COLORS
from .recorder import Recorder
from .agent import Direction

# ── Rendu ──────────────────────────────────────────────────────────────────
TILE_SIZE     = 20          # pixels par case
HEADER_HEIGHT = 40          # pixels réservés en haut pour score/vies
FPS           = 10          # ticks de jeu par seconde

# Palette
C = {
    "bg":             (  0,   0,   0),
    "wall":           ( 33,  33, 222),
    "ghost_wall":     (255, 182, 193),
    "pacman":         (255, 255,   0),
    "pellet":         (255, 255, 255),
    "power_pellet":   (255, 255, 255),
    "frightened":     (  0,   0, 200),
    "text":           (255, 255, 255),
    "text_yellow":    (255, 255,   0),
    "header_bg":      (  0,   0,   0),
}

# Score à partir duquel on offre une vie bonus
BONUS_LIFE_THRESHOLD = 10_000


class Game:
    """
    Boucle principale du jeu Pac-Man.

    Architecture extensible
    -----------------------
    - Pour remplacer le contrôle clavier par une IA Pac-Man :
        Passer une instance de sous-classe PacManAgent à Game (sans rien changer ici).
    - Pour brancher une IA sur un fantôme :
        Remplacer ghosts[i] par une instance de sous-classe Ghost (voir ghost.py).

    Paramètres
    ----------
    seed   : int | None   — seed pour le générateur de labyrinthe
    local  : bool         — utiliser localhost:5000 au lieu de Render
    record : bool         — activer le Recorder (sauvegarde recording.json)
    pacman_agent : PacManAgent | None — agent personnalisé (clavier par défaut)
    """

    def __init__(
        self,
        seed=None,
        local: bool = False,
        record: bool = False,
        pacman_agent: PacManAgent | None = None,
    ):
        self.base_url      = LOCAL_API_URL if local else CLOUD_API_URL
        self.seed          = seed
        self.record        = record
        self._pacman_agent = pacman_agent   # None → instancié dans run()
        self._bonus_given  = False

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------

    def run(self) -> None:
        pygame.init()

        # ── Environnement ──────────────────────────────────────────────
        env = Environment(seed=self.seed, base_url=self.base_url)
        print("Chargement du labyrinthe depuis l'API…")
        try:
            env.load_maze()
        except Exception as exc:
            print(f"Erreur API : {exc}")
            pygame.quit()
            sys.exit(1)
        print(f"Labyrinthe chargé — seed: {env.seed}  ({env.width}×{env.height})")

        # ── Agents ────────────────────────────────────────────────────
        pacman = self._pacman_agent or PacManAgent()
        ghosts = [Ghost(i, gx, gy) for i, (gx, gy) in enumerate(GHOST_START_POSITIONS)]
        env.ghosts = ghosts

        # ── Recorder ──────────────────────────────────────────────────
        recorder = Recorder(seed=env.seed) if self.record else None

        # ── Fenêtre pygame ────────────────────────────────────────────
        win_w = env.width  * TILE_SIZE
        win_h = env.height * TILE_SIZE + HEADER_HEIGHT
        screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption(f"Pac-Man  —  seed {env.seed}")
        clock = pygame.time.Clock()
        font     = pygame.font.SysFont("monospace", 15, bold=True)
        font_big = pygame.font.SysFont("monospace", 30, bold=True)

        paused  = False
        running = True

        while running:
            clock.tick(FPS)

            # ── Événements pygame ──────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    if event.key == pygame.K_p:
                        paused = not paused
                    # R = rejouer (uniquement sur l'écran game over)
                    if event.key == pygame.K_r and env.game_over:
                        env = Environment(seed=self.seed, base_url=self.base_url)
                        env.load_maze()
                        ghosts = [Ghost(i, gx, gy)
                                  for i, (gx, gy) in enumerate(GHOST_START_POSITIONS)]
                        env.ghosts = ghosts
                        pacman = self._pacman_agent or PacManAgent()
                        self._bonus_given = False
                        if self.record:
                            recorder = Recorder(seed=env.seed)
                        pygame.display.set_caption(f"Pac-Man  —  seed {env.seed}")
                    pacman.handle_event(event)

            # ── Écrans d'état ──────────────────────────────────────────
            if env.game_over:
                self._draw(screen, env, font)
                msg  = font_big.render("GAME OVER", True, (255, 0, 0))
                hint = font.render("R = rejouer  |  Q = quitter", True, C["text"])
                screen.blit(msg,  (win_w // 2 - msg.get_width()  // 2,
                                   win_h // 2 - msg.get_height() // 2))
                screen.blit(hint, (win_w // 2 - hint.get_width() // 2,
                                   win_h // 2 + msg.get_height()))
                pygame.display.flip()
                continue

            if paused:
                self._draw(screen, env, font)
                msg = font_big.render("PAUSE", True, C["text_yellow"])
                screen.blit(msg, (win_w // 2 - msg.get_width() // 2,
                                  win_h // 2 - msg.get_height() // 2))
                pygame.display.flip()
                continue

            # ── Passage au niveau suivant ──────────────────────────────
            if env.level_complete:
                env.next_level()
                self._bonus_given = False   # reset pour le nouveau niveau
                continue

            # ── Collecte des actions ───────────────────────────────────
            state = env.get_state()
            actions = {"pacman": pacman.get_action(state)}
            for i, ghost in enumerate(ghosts):
                actions[f"ghost_{i}"] = ghost.get_action(state)

            # ── Mise à jour du monde ───────────────────────────────────
            events = env.update(actions)

            # Confirmer la direction effective à Pac-Man
            pacman.confirm_direction(env.pacman_dir)

            # Vie bonus à 10 000 pts
            if not self._bonus_given and env.score >= BONUS_LIFE_THRESHOLD:
                env.lives += 1
                self._bonus_given = True
                print("[Game] Vie bonus accordée !")

            # ── Enregistrement ────────────────────────────────────────
            if recorder:
                recorder.record(actions, env.get_grid_snapshot())

            # ── Rendu ─────────────────────────────────────────────────
            self._draw(screen, env, font)
            pygame.display.flip()

        # ── Fin ───────────────────────────────────────────────────────
        if recorder:
            recorder.save("recording.json")

        pygame.quit()
        sys.exit(0)

    # ------------------------------------------------------------------
    # Rendu
    # ------------------------------------------------------------------

    def _draw(self, screen: pygame.Surface, env: Environment,
              font: pygame.font.Font) -> None:
        screen.fill(C["bg"])
        oy = HEADER_HEIGHT   # offset vertical pour le plateau

        # ── Murs ──────────────────────────────────────────────────────
        for (x, y) in env.walls:
            pygame.draw.rect(screen, C["wall"],
                             (x * TILE_SIZE, y * TILE_SIZE + oy,
                              TILE_SIZE, TILE_SIZE))

        # ── Porte de la maison des fantômes ───────────────────────────
        for (x, y) in env.ghost_walls:
            mid_y = y * TILE_SIZE + oy + TILE_SIZE // 2 - 1
            pygame.draw.rect(screen, C["ghost_wall"],
                             (x * TILE_SIZE, mid_y, TILE_SIZE, 3))

        # ── Pellets ───────────────────────────────────────────────────
        for (px, py), pellet in env.pellets.items():
            cx = px * TILE_SIZE + TILE_SIZE // 2
            cy = py * TILE_SIZE + TILE_SIZE // 2 + oy
            if pellet.type == "power":
                pygame.draw.circle(screen, C["power_pellet"], (cx, cy),
                                   TILE_SIZE // 4)
            else:
                pygame.draw.circle(screen, C["pellet"], (cx, cy), 2)

        # ── Fantômes ──────────────────────────────────────────────────
        for i, ghost in enumerate(env.ghosts):
            cx = ghost.x * TILE_SIZE + TILE_SIZE // 2
            cy = ghost.y * TILE_SIZE + TILE_SIZE // 2 + oy
            color = C["frightened"] if ghost.frightened \
                else GHOST_COLORS[i % len(GHOST_COLORS)]
            # Corps (demi-cercle + rectangle)
            r = TILE_SIZE // 2 - 1
            pygame.draw.circle(screen, color, (cx, cy - r // 2), r)
            pygame.draw.rect(screen, color,
                             (cx - r, cy - r // 2, r * 2, r + 1))

        # ── Pac-Man ───────────────────────────────────────────────────
        cx = env.pacman_x * TILE_SIZE + TILE_SIZE // 2
        cy = env.pacman_y * TILE_SIZE + TILE_SIZE // 2 + oy
        pygame.draw.circle(screen, C["pacman"], (cx, cy), TILE_SIZE // 2 - 1)

        # ── En-tête (score / vies / niveau) ───────────────────────────
        pygame.draw.rect(screen, C["header_bg"],
                         (0, 0, screen.get_width(), HEADER_HEIGHT))

        score_s = font.render(f"Score: {env.score}", True, C["text"])
        lives_s = font.render(f"Vies: {env.lives}", True, C["text"])
        level_s = font.render(f"Niveau: {env.level}", True, C["text_yellow"])

        screen.blit(score_s, (10,  12))
        screen.blit(lives_s, (200, 12))
        screen.blit(level_s, (370, 12))

        if env.frightened_timer > 0:
            ft = font.render(f"Frightened: {env.frightened_timer}", True,
                             C["frightened"])
            screen.blit(ft, (500, 12))
