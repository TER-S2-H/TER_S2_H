"""
RL avec module de coût placé directement après le prédicteur de niveau 1.
Pas de planification hiérarchique : pour chaque action possible (0-3),
on prédit e_{t+1} avec predictor_1, on évalue le coût, et on choisit
l'action qui minimise ce coût.

Le buffer de backprop reste de taille j (horizon de coût réel identique
à rl_loop.py), seul le mécanisme de sélection d'action change.

Usage :
    python -m src.jepacman.rl_loop_level1 \\
        --maze-dir     generated_mazes \\
        --jepa-weights optuna_results/best_model.pt \\
        --jepa-config  optuna_results/best_config_full.json \\
        --n-episodes   200 \\
        --lr           1e-4
"""

import os
import sys
import json
import glob
import random
import argparse
import dataclasses
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import itertools

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .models.jepa import HierarchicalJEPAcman
from .models.cost_module import CostModule

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from ..visualize_session import draw_frame, draw_static_maze
from .convert_sess_to_jepa import (
    compute_visible_coords,
    build_cell_dict,
    PAD_CELL,
    MAX_VISIBLE_CELLS,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.pacman_simulator import (
    load_maze, build_graph, extract_walkable_and_spawns,
    symmetrize_energizers, GameState, step as sim_step,
    manhattan,
)

Coord = Tuple[int, int]

OUTPUT_DIR  = "optuna_results"
LR_DEFAULT  = 1e-4
REVERSE_MAP = {0: 1, 1: 0, 2: 3, 3: 2}
CHAIN_LEN   = 5   # horizon de la chaîne de predictor_1


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_coord_adj(state: GameState) -> Dict[Coord, Set[Coord]]:
    adj: Dict[Coord, Set[Coord]] = {}
    for idx, neighbors in state.adjacency.items():
        coord = state.index_to_coord[idx]
        adj[coord] = {state.index_to_coord[n] for n in neighbors}
    return adj


def build_obs_tensors(
    agent_coord:       Coord,
    adj:               Dict[Coord, Set[Coord]],
    active_pellets:    Set[Coord],
    active_energizers: Set[Coord],
    ghost_coords:      List[Coord],
    pacman_coord:      Coord,
    d_model:           int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    visible = compute_visible_coords(agent_coord, adj)
    cells   = [
        build_cell_dict(c, agent_coord, adj,
                        active_pellets, active_energizers,
                        ghost_coords, pacman_coord)
        for c in visible
    ]
    n_real = len(cells)
    if n_real < MAX_VISIBLE_CELLS:
        cells += [PAD_CELL] * (MAX_VISIBLE_CELLS - n_real)
    else:
        cells = cells[:MAX_VISIBLE_CELLS]

    cases = np.array([[
        c["wall_north"], c["wall_south"], c["wall_east"], c["wall_west"],
        c["has_pellet"], c["has_super_pellet"], c["has_ghost"], c["has_pacman"],
    ] for c in cells], dtype=np.float32)

    positions = np.array([[c["dx"], c["dy"]] for c in cells], dtype=np.float32)
    pad_mask  = np.array([c["is_pad"] for c in cells], dtype=bool)

    return cases, positions, pad_mask


# ── Agent JEPA niveau 1 ───────────────────────────────────────────────────────

class JEPAAgentLevel1:
    """
    Agent qui sélectionne son action en évaluant directement le coût sur
    e_{t+1} = predictor_1(seq, a) pour chaque action a ∈ {0,1,2,3}.

    Le buffer de backprop est identique à rl_loop.py (taille j) :
    la prédiction stockée est (seq, e_next_chosen, g_dummy=0)
    et le coût réel est calculé j pas plus tard.
    """

    def __init__(
        self,
        agent_id:    int,
        jepa:        HierarchicalJEPAcman,
        cost_module: CostModule,
        config:      Config,
        device:      torch.device,
    ):
        self.agent_id    = agent_id
        self.is_pacman   = (agent_id == 0)
        self.jepa        = jepa
        self.cost_module = cost_module
        self.config      = config
        self.device      = device
        self.last_action = None          # pour anti-retour-arrière

        self.context: deque = deque(maxlen=config.seq_len)
        self.buffer_preds: deque = deque()

        self.total_loss = 0.0
        self.n_updates  = 0

    def reset(self):
        self.context.clear()
        self.buffer_preds.clear()
        self.last_action = None

    def observe(self, cases: np.ndarray, pos: np.ndarray,
                mask: np.ndarray, action: int):
        self.context.append((cases, pos, mask, action))

    def _build_seq_tensor(self) -> torch.Tensor:
        seq_len = self.config.seq_len
        n_ctx   = len(self.context)
        n_pad   = seq_len - n_ctx

        def make_tensor(key):
            items = list(self.context)
            if key == "cases":
                arr = np.stack([x[0] for x in items])
            elif key == "pos":
                arr = np.stack([x[1] for x in items])
            elif key == "mask":
                arr = np.stack([x[2] for x in items])
            else:  # action
                arr = np.array([x[3] for x in items], dtype=np.int64)
                arr = np.where(arr == -1, 4, arr)
                if n_pad > 0:
                    return torch.from_numpy(
                        np.concatenate([np.full(n_pad, 5, dtype=np.int64), arr])
                    ).unsqueeze(0).to(self.device)
                return torch.from_numpy(arr).unsqueeze(0).to(self.device)
            if n_pad > 0:
                pad_val   = True if key == "mask" else 0.0
                pad_shape = (n_pad,) + arr.shape[1:]
                pad = np.full(pad_shape, pad_val, dtype=arr.dtype)
                arr = np.concatenate([pad, arr], axis=0)
            return torch.from_numpy(arr).unsqueeze(0).to(self.device)

        cases_t   = make_tensor("cases")
        pos_t     = make_tensor("pos")
        masks_t   = make_tensor("mask")
        actions_t = make_tensor("action")

        with torch.no_grad():
            seq = self.jepa.encode_sequence(
                cases_t, pos_t, actions_t, masks_t
            )   # [1, T, d_model]
        return seq

    def choose_action(self) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Énumère toutes les séquences de CHAIN_LEN actions (4^CHAIN_LEN = 1024),
        chaîne CHAIN_LEN appels à predictor_1 pour chacune, évalue le coût sur
        l'état final prédit, et exécute la première action de la meilleure séquence.

        Variable latente → code TEM de la séquence d'actions : tem(a0..a4)[:, -1, :]
        Retourne (first_action, seq, e_final, g_chain).
        """
        if len(self.context) == 0:
            return random.randint(0, 3), None, None, None

        seq = self._build_seq_tensor()   # [1, T, d_model]

        forbidden_first: Set[int] = set()
        if self.last_action is not None:
            forbidden_first = {REVERSE_MAP[self.last_action]}

        best_first_action = None
        best_cost         = float("inf")
        best_e_final      = None
        best_g_chain      = None

        with torch.no_grad():
            # Toutes les séquences valides (première action non interdite)
            all_seqs   = list(itertools.product(range(4), repeat=CHAIN_LEN))
            valid_seqs = [s for s in all_seqs if s[0] not in forbidden_first]
            if not valid_seqs:
                valid_seqs = all_seqs   # fallback

            N = len(valid_seqs)   # ≤ 4^CHAIN_LEN

            # Tensor de toutes les séquences : [N, CHAIN_LEN]
            actions_batch = torch.tensor(
                valid_seqs, dtype=torch.long, device=self.device
            )

            # Expand seq pour le batch : [N, T, d_model]
            current_seq = seq.expand(N, -1, -1).contiguous()
            seq_len     = self.jepa.seq_encoder.seq_len

            # CHAIN_LEN forward passes batché (1 pass = N séquences en parallèle)
            e_step = None
            for t in range(CHAIN_LEN):
                acts_t = actions_batch[:, t]                            # [N]
                g_a    = self.jepa.predictor_1.action_embed(acts_t)    # [N, d_model]
                e_step = self.jepa.predictor_1(current_seq, acts_t)    # [N, d_model]
                token  = (e_step + g_a).unsqueeze(1)                   # [N, 1, d_model]
                current_seq = torch.cat([current_seq, token], dim=1)   # [N, T+t+1, d_model]
                current_seq = current_seq[:, -seq_len:]                 # tronquer

            # Latent TEM pour toutes les séquences : [N, d_model]
            g_chains = self.jepa.tem(actions_batch)[:, -1, :]

            # Coût pour toutes les séquences en un seul forward pass
            seq_exp = seq.expand(N, -1, -1)                            # [N, T, d_model]
            costs   = self.cost_module(seq_exp, e_step, g_chains)      # [N]

            best_idx          = costs.argmin().item()
            best_first_action = valid_seqs[best_idx][0]
            best_e_final      = e_step[best_idx : best_idx + 1]        # [1, d_model]
            best_g_chain      = g_chains[best_idx : best_idx + 1]      # [1, d_model]

        return best_first_action, seq, best_e_final, best_g_chain

    def store_prediction(
        self,
        seq:         Optional[torch.Tensor],
        e_next:      Optional[torch.Tensor],
        g_latent:    Optional[torch.Tensor],
        n_pellets_t: int,
        pac_xy_t:    Coord,
        ghost_xy_t:  List[Coord],
    ):
        if seq is None:
            return   # contexte vide au tout début : pas de prédiction à stocker
        self.buffer_preds.append((
            seq.detach().clone(),
            e_next.detach().clone(),
            g_latent.detach().clone(),
            n_pellets_t, pac_xy_t, ghost_xy_t,
        ))

    def maybe_backprop(
        self,
        n_pellets_now:  int,
        pac_xy_now:     Coord,
        ghost_xy_now:   List[Coord],
        outcome_eaten:  bool,
        optimizer:      torch.optim.Optimizer,
    ) -> Optional[float]:
        # Buffer de taille CHAIN_LEN : on backpropage quand on a l'état t+CHAIN_LEN
        if len(self.buffer_preds) < CHAIN_LEN:
            return None

        seq, e_next, g_latent, n_pellets_t, pac_xy_t, ghost_xy_t = \
            self.buffer_preds.popleft()

        # Recalcul du coût prédit (graphe frais pour backprop)
        cost_pred = self.cost_module(seq, e_next, g_latent)

        # Coût réel sur CHAIN_LEN pas
        if self.is_pacman:
            died = outcome_eaten and len(self.buffer_preds) == 0
            if died:
                real_cost = float(CHAIN_LEN * 10)
            else:
                eaten     = n_pellets_t - n_pellets_now
                real_cost = float(CHAIN_LEN - eaten)
        else:
            ghost_idx = self.agent_id - 1
            gx_t, gy_t = ghost_xy_t[ghost_idx]
            px_t, py_t = pac_xy_t
            gx_n, gy_n = ghost_xy_now[ghost_idx]
            px_n, py_n = pac_xy_now
            dist_t  = abs(gx_t - px_t) + abs(gy_t - py_t)
            dist_t5 = abs(gx_n - px_n) + abs(gy_n - py_n)
            real_cost = float(2 * dist_t5 - dist_t)

        target = torch.tensor([real_cost], dtype=torch.float32, device=self.device)
        loss   = F.huber_loss(cost_pred, target, delta=1.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val = loss.item()
        self.total_loss += val
        self.n_updates  += 1
        return val

    def flush_buffer(
        self,
        history:       List,
        outcome_eaten: bool,
        optimizer:     torch.optim.Optimizer,
    ):
        offset = 0
        while self.buffer_preds:
            seq, e_next, g_latent, n_pellets_t, pac_xy_t, ghost_xy_t = \
                self.buffer_preds.popleft()

            cost_pred = self.cost_module(seq, e_next, g_latent)
            snap_idx  = min(offset + CHAIN_LEN, len(history) - 1)
            n_pellets_now, pac_xy_now, ghost_xy_now = history[snap_idx]

            if self.is_pacman:
                died      = outcome_eaten and snap_idx == len(history) - 1
                real_cost = (float(CHAIN_LEN * 10) if died
                             else float(CHAIN_LEN - (n_pellets_t - n_pellets_now)))
            else:
                ghost_idx = self.agent_id - 1
                gx_t, gy_t = ghost_xy_t[ghost_idx]
                px_t, py_t = pac_xy_t
                gx_n, gy_n = ghost_xy_now[ghost_idx]
                px_n, py_n = pac_xy_now
                dist_t  = abs(gx_t - px_t) + abs(gy_t - py_t)
                dist_t5 = abs(gx_n - px_n) + abs(gy_n - py_n)
                real_cost = float(2 * dist_t5 - dist_t)

            target = torch.tensor([real_cost], dtype=torch.float32, device=self.device)
            loss      = F.huber_loss(cost_pred, target, delta=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.total_loss += loss.item()
            self.n_updates  += 1
            offset += 1


# ── Boucle RL principale ──────────────────────────────────────────────────────

def run_rl(
    maze_dir:      str,
    jepa_weights:  str,
    jepa_config:   str,
    n_episodes:    int   = 200,
    lr:            float = LR_DEFAULT,
    save_every:    int   = 50,
    render:        bool  = False,
    render_every:  int   = 10,
    render_fps:    int   = 12,
):
    device = get_device()
    print(f"Dispositif : {device}")

    # ── Charger JEPA (gelé) ───────────────────────────────────────────────────
    with open(jepa_config) as f:
        cfg_raw = json.load(f)
    valid  = {f.name for f in dataclasses.fields(Config)}
    config = Config(**{k: v for k, v in cfg_raw.items() if k in valid})

    jepa = HierarchicalJEPAcman(
        case_dim=config.case_dim, n_actions=config.n_actions,
        d_model=config.d_model, seq_len=config.seq_len,
        n_heads_state=config.n_heads_state, n_layers_state=config.n_layers_state,
        n_heads_seq=config.n_heads_seq, n_layers_seq=config.n_layers_seq,
        n_heads_pred_1=config.n_heads_pred_1, n_layers_pred_1=config.n_layers_pred_1,
        n_heads_pred_2=config.n_heads_pred_2, n_layers_pred_2=config.n_layers_pred_2,
        n_heads_pred_3=config.n_heads_pred_3, n_layers_pred_3=config.n_layers_pred_3,
        ema_momentum=config.ema_momentum, k=config.k, j=config.j,
    )
    jepa.load_state_dict(torch.load(jepa_weights, map_location="cpu"))
    jepa.to(device).eval()
    for p in jepa.parameters():
        p.requires_grad = False

    # ── Charger les modules de coût ───────────────────────────────────────────
    # Architecture : cherche d'abord les params spécifiques L1, sinon params généraux
    cost_config_path = os.path.join(OUTPUT_DIR, "best_params_cost_modules_l1.json")
    if not os.path.exists(cost_config_path):
        cost_config_path = os.path.join(OUTPUT_DIR, "best_params_cost_modules.json")
    if os.path.exists(cost_config_path):
        with open(cost_config_path) as f:
            cost_params = json.load(f)
        cost_n_heads  = cost_params.get("n_heads",  2)
        cost_n_layers = cost_params.get("n_layers", 2)
        print(f"Architecture cost modules : n_heads={cost_n_heads}, n_layers={cost_n_layers}")
    else:
        cost_n_heads, cost_n_layers = 2, 2
        print("⚠ best_params_cost_modules_l1.json introuvable — architecture par défaut")

    cost_pacman = CostModule(config.d_model, cost_n_heads, cost_n_layers).to(device)
    cost_ghost  = CostModule(config.d_model, cost_n_heads, cost_n_layers).to(device)

    # Poids : cherche d'abord les poids spécifiques L1, sinon poids généraux
    pac_weights_path   = os.path.join(OUTPUT_DIR, "cost_pacman_l1.pt")
    ghost_weights_path = os.path.join(OUTPUT_DIR, "cost_ghost_l1.pt")
    if not os.path.exists(pac_weights_path):
        pac_weights_path = os.path.join(OUTPUT_DIR, "cost_pacman.pt")
    if not os.path.exists(ghost_weights_path):
        ghost_weights_path = os.path.join(OUTPUT_DIR, "cost_ghost.pt")
    if os.path.exists(pac_weights_path):
        cost_pacman.load_state_dict(torch.load(pac_weights_path, map_location="cpu"))
        print(f"cost_pacman chargé depuis {pac_weights_path}")
    if os.path.exists(ghost_weights_path):
        cost_ghost.load_state_dict(torch.load(ghost_weights_path, map_location="cpu"))
        print(f"cost_ghost chargé depuis {ghost_weights_path}")

    optimizer = torch.optim.Adam(
        list(cost_pacman.parameters()) + list(cost_ghost.parameters()), lr=lr
    )

    # ── Lister les labyrinthes ────────────────────────────────────────────────
    maze_files = sorted(glob.glob(os.path.join(maze_dir, "**", "maze.json"), recursive=True))
    if not maze_files:
        raise FileNotFoundError(f"Aucun maze.json trouvé dans {maze_dir}")
    print(f"{len(maze_files)} labyrinthes disponibles")

    # ── Créer les agents ──────────────────────────────────────────────────────
    agents = [
        JEPAAgentLevel1(0, jepa, cost_pacman, config, device),   # Pac-Man
        JEPAAgentLevel1(1, jepa, cost_ghost,  config, device),   # Ghost 0
        JEPAAgentLevel1(2, jepa, cost_ghost,  config, device),   # Ghost 1
        JEPAAgentLevel1(3, jepa, cost_ghost,  config, device),   # Ghost 2
        JEPAAgentLevel1(4, jepa, cost_ghost,  config, device),   # Ghost 3
    ]

    screen, clock, font, tile_size = None, None, None, 22
    if render:
        if not PYGAME_AVAILABLE:
            print("⚠ pygame non disponible — rendu désactivé")
            render = False
        else:
            pygame.init()
            clock = pygame.time.Clock()
            pygame.display.set_caption("JEPAcman RL — Level-1")

    # ── Boucle épisodes ───────────────────────────────────────────────────────
    for ep in range(n_episodes):
        maze_json_path = random.choice(maze_files)
        maze_data      = load_maze(maze_json_path)
        meta   = maze_data.get("metadata", {})
        width  = meta.get("width",  28)
        height = meta.get("height", 31)

        do_render = render and (ep % render_every == 0)
        if do_render:
            with open(maze_json_path) as f:
                maze_json_data = json.load(f)
            screen = pygame.display.set_mode(
                (width * tile_size, height * tile_size + 40)
            )
            font = pygame.font.SysFont(None, 22)

        coo_src, coo_dst, node_to_index, index_to_coord, adjacency = build_graph(
            maze_data["nodes"], maze_data["edges"], width, height
        )
        energizers_full = symmetrize_energizers(
            maze_data["energizers"], node_to_index, height
        )
        pacman_spawn, ghost_spawns = extract_walkable_and_spawns(
            maze_data["nodes"], node_to_index, maze_data.get("ghost_walls", set())
        )

        state = GameState(
            adjacency         = adjacency,
            node_to_index     = node_to_index,
            index_to_coord    = index_to_coord,
            ghost_wall_nodes  = set(),
            pacman_spawn      = pacman_spawn,
            ghost_spawns      = ghost_spawns,
            initial_pellets   = set(
                node_to_index[tuple(p)]
                for p in maze_data.get("pellets", set())
                if tuple(p) in node_to_index
            ),
            initial_energizers = set(
                node_to_index[e]
                for e in energizers_full
                if e in node_to_index
            ),
        )

        coord_adj = build_coord_adj(state)
        for agent in agents:
            agent.reset()

        history       = []   # [(n_pellets, pac_xy, ghost_xy_list), ...]
        outcome_eaten = False
        step          = 0
        done          = False

        while not done:
            pac_coord    = state.index_to_coord[state.pacman_pos]
            ghost_coords = [
                state.index_to_coord[state.ghost_positions[g]]
                for g in range(4)
            ]
            active_pellets    = {state.index_to_coord[p] for p in state.active_pellets}
            active_energizers = {state.index_to_coord[e] for e in state.active_energizers}

            n_pellets_now = len(state.active_pellets) + len(state.active_energizers)
            history.append((n_pellets_now, pac_coord, list(ghost_coords)))

            actions_chosen = {}

            for agent in agents:
                if agent.is_pacman:
                    agent_coord = pac_coord
                    gc, pc      = ghost_coords, pac_coord
                else:
                    agent_coord = ghost_coords[agent.agent_id - 1]
                    gc, pc      = ghost_coords, pac_coord

                cases, pos, mask = build_obs_tensors(
                    agent_coord, coord_adj,
                    active_pellets, active_energizers,
                    gc, pc, config.d_model,
                )

                prev_action = int(agent.context[-1][3]) if agent.context else -1
                agent.observe(cases, pos, mask, prev_action)

                # ── Sélection d'action via predictor_1 + cost_module ──────
                action_idx, seq, e_next, g_latent = agent.choose_action()
                agent.last_action = action_idx

                agent.store_prediction(
                    seq, e_next, g_latent,
                    n_pellets_now, pac_coord, list(ghost_coords),
                )

                if agent.is_pacman:
                    actions_chosen["pacman"] = action_idx
                else:
                    actions_chosen[f"ghost_{agent.agent_id - 1}"] = action_idx

            # ── Backprop des prédictions matures ──────────────────────────
            for agent in agents:
                agent.maybe_backprop(
                    n_pellets_now, pac_coord, list(ghost_coords),
                    outcome_eaten=False, optimizer=optimizer,
                )

            done, outcome = sim_step(
                state,
                is_first_step = (step == 0),
                pacman_action = actions_chosen.get("pacman"),
                ghost_actions = [actions_chosen.get(f"ghost_{i}") for i in range(4)],
            )

            # ── Rendu ─────────────────────────────────────────────────────
            if do_render and state.frames:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                frame     = state.frames[-1]
                maze_surf = screen.subsurface(
                    pygame.Rect(0, 40, width * tile_size, height * tile_size)
                )
                draw_frame(maze_surf, frame, maze_json_data, tile_size)

                pygame.draw.rect(screen, (12, 12, 12),
                                 pygame.Rect(0, 0, width * tile_size, 40))
                hud = font.render(
                    f"Ep {ep+1}/{n_episodes} | step {step} | "
                    f"loss_pac={agents[0].total_loss/max(agents[0].n_updates,1):.3f} "
                    f"[Level-1 only]",
                    True, (255, 255, 255),
                )
                screen.blit(hud, (8, 12))
                pygame.display.flip()
                clock.tick(render_fps)

            if outcome == "eaten":
                outcome_eaten = True
            step += 1

        # ── Fin d'épisode ────────────────────────────────────────────────────
        for agent in agents:
            agent.flush_buffer(history, outcome_eaten, optimizer)

        pac_loss = agents[0].total_loss / max(agents[0].n_updates, 1)
        gh_loss  = (sum(a.total_loss for a in agents[1:])
                    / max(sum(a.n_updates for a in agents[1:]), 1))
        print(
            f"Ep {ep+1:4d}/{n_episodes} | steps={step:4d} | "
            f"outcome={outcome} | "
            f"loss_pac={pac_loss:.4f}  loss_ghost={gh_loss:.4f}"
        )

        for agent in agents:
            agent.total_loss = 0.0
            agent.n_updates  = 0

        # ── Sauvegarde périodique ─────────────────────────────────────────────
        if (ep + 1) % save_every == 0:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            torch.save(cost_pacman.state_dict(),
                       os.path.join(OUTPUT_DIR, "cost_pacman_rl_l1.pt"))
            torch.save(cost_ghost.state_dict(),
                       os.path.join(OUTPUT_DIR, "cost_ghost_rl_l1.pt"))
            print(f"  ✓ Poids RL level-1 sauvegardés (ep {ep+1})")

    print("\nRL (level-1) terminé.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(cost_pacman.state_dict(),
               os.path.join(OUTPUT_DIR, "cost_pacman_rl_l1.pt"))
    torch.save(cost_ghost.state_dict(),
               os.path.join(OUTPUT_DIR, "cost_ghost_rl_l1.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RL avec module de coût après predictor_1 uniquement"
    )
    parser.add_argument("--maze-dir",      default="generated_mazes")
    parser.add_argument("--jepa-weights",  required=True)
    parser.add_argument("--jepa-config",   required=True)
    parser.add_argument("--n-episodes",    type=int,   default=200)
    parser.add_argument("--lr",            type=float, default=LR_DEFAULT)
    parser.add_argument("--save-every",    type=int,   default=50)
    parser.add_argument("--render",        action="store_true")
    parser.add_argument("--render-every",  type=int,   default=10)
    parser.add_argument("--render-fps",    type=int,   default=12)
    args = parser.parse_args()

    run_rl(
        maze_dir      = args.maze_dir,
        jepa_weights  = args.jepa_weights,
        jepa_config   = args.jepa_config,
        n_episodes    = args.n_episodes,
        lr            = args.lr,
        save_every    = args.save_every,
        render        = args.render,
        render_every  = args.render_every,
        render_fps    = args.render_fps,
    )
