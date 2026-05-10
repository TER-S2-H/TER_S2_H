"""
Phase de renforcement des modules de coût JEPA.

Pac-Man et les fantômes jouent en utilisant les modèles JEPA entraînés.
À chaque pas t, le module de coût prédit le coût pour t+j (avec gradient).
Quand on atteint t+j, on calcule le coût réel et on backpropage.

Principe du buffer de prédictions :
    buffer[i] = (cost_pred_tensor, game_snapshot_at_t)
    Taille fixe = j.
    Quand le buffer est plein, pop(0) → backprop → push(new_pred).

Usage :
    python -m src.jepacman.rl_loop \\
        --maze-json   path/to/maze.json \\
        --jepa-weights optuna_results/best_model.pt \\
        --jepa-config  optuna_results/best_config_full.json \\
        --n-episodes  200 \\
        --lr          1e-4
"""

import os
import sys
import json
import argparse
import dataclasses
import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .models.jepa import HierarchicalJEPAcman
from .models.cost_module import CostModule

# Fonctions de visibilité/observation réutilisées depuis convert_sess_to_jepa
from .convert_sess_to_jepa import (
    compute_visible_coords,
    build_cell_dict,
    PAD_CELL,
    MAX_VISIBLE_CELLS,
)

# Import du moteur de jeu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.pacman_simulator import (
    load_maze, build_graph, extract_walkable_and_spawns,
    symmetrize_energizers, GameState, step as sim_step,
    manhattan,
)

Coord = Tuple[int, int]

OUTPUT_DIR = "optuna_results"
LR_DEFAULT = 1e-4


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_coord_adj(state: GameState) -> Dict[Coord, Set[Coord]]:
    """Reconstruit l'adjacence coord→coords depuis la GameState."""
    adj: Dict[Coord, Set[Coord]] = {}
    for idx, neighbors in state.adjacency.items():
        coord = state.index_to_coord[idx]
        adj[coord] = {state.index_to_coord[n] for n in neighbors}
    return adj


def build_obs_tensors(
    agent_coord: Coord,
    adj: Dict[Coord, Set[Coord]],
    active_pellets: Set[Coord],
    active_energizers: Set[Coord],
    ghost_coords: List[Coord],
    pacman_coord: Coord,
    d_model: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construit les tableaux numpy pour une observation agent.
    Retourne (cases [81,8], positions [81,2], pad_mask [81]).
    """
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


# ── Optimisation de g_j avec le module de coût ────────────────────────────────

def plan_with_cost(
    jepa:        HierarchicalJEPAcman,
    cost_module: CostModule,
    seq:         torch.Tensor,   # [1, T, d_model]
    n_steps:     int = 30,
    lr:          float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimise g_j pour minimiser le coût prédit.
    Retourne (g_j_opt [1, d_model], e_goal [1, d_model]).
    JEPA gelé ; seul g_j est mis à jour.
    """
    d = seq.shape[-1]
    g_j = torch.randn(1, d, requires_grad=True, device=seq.device)
    opt = torch.optim.Adam([g_j], lr=lr)

    for _ in range(n_steps):
        opt.zero_grad()
        e_pred = jepa.predictor_3.forward_with_latent(seq.detach(), g_j)
        cost   = cost_module(seq.detach(), e_pred, g_j).mean()
        cost.backward()
        opt.step()

    g_j = g_j.detach()
    with torch.no_grad():
        e_goal = jepa.predictor_3.forward_with_latent(seq.detach(), g_j)
    return g_j, e_goal


# ── Agent JEPA ────────────────────────────────────────────────────────────────

class JEPAAgent:
    """
    Agent qui maintient un contexte glissant et prédit les coûts online.

    buffer_preds : deque de (cost_pred_tensor, n_pellets_t, pac_xy_t, ghost_xy_t)
    Taille max = j. Quand plein → backprop sur la prédiction la plus ancienne.
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

        # Contexte glissant : derniers seq_len observations
        # Chaque entrée : (cases [81,8], pos [81,2], mask [81], action int)
        self.context: deque = deque(maxlen=config.seq_len)

        # Buffer des prédictions en attente de leur coût réel
        # Chaque entrée : (cost_pred, n_pellets_at_t, pac_xy_at_t, ghost_xy_at_t)
        self.buffer_preds: deque = deque()

        # Cumul de loss pour log
        self.total_loss  = 0.0
        self.n_updates   = 0

    def reset(self):
        self.context.clear()
        self.buffer_preds.clear()

    def observe(self, cases: np.ndarray, pos: np.ndarray,
                mask: np.ndarray, action: int):
        """Ajoute une observation au contexte."""
        self.context.append((cases, pos, mask, action))

    def _build_seq_tensor(self) -> torch.Tensor:
        """Construit le tenseur [1, T, d_model] depuis le contexte."""
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
                pad_val = True if key == "mask" else 0.0
                pad_shape = (n_pad,) + arr.shape[1:]
                pad = np.full(pad_shape, pad_val, dtype=arr.dtype)
                arr = np.concatenate([pad, arr], axis=0)
            return torch.from_numpy(arr).unsqueeze(0).to(self.device)

        cases_t   = make_tensor("cases")    # [1, T, 81, 8]
        pos_t     = make_tensor("pos")      # [1, T, 81, 2]
        masks_t   = make_tensor("mask")     # [1, T, 81]
        actions_t = make_tensor("action")   # [1, T]

        with torch.no_grad():
            seq = self.jepa.encode_sequence(
                cases_t, pos_t, actions_t, masks_t
            )  # [1, T, d_model]
        return seq

    def choose_action(self) -> int:
        """
        Planifie avec JEPA + module de coût et retourne l'action choisie (0-3).
        Stocke la prédiction de coût dans le buffer.
        """
        if len(self.context) == 0:
            return random.randint(0, 3)

        seq = self._build_seq_tensor()  # [1, T, d_model]

        # 1. Optimiser g_j avec le module de coût (gradient sur cost_module)
        g_j, e_goal = plan_with_cost(
            self.jepa, self.cost_module, seq,
            n_steps=20, lr=0.05,
        )

        # 2. Prédire et stocker le coût (avec graphe de calcul pour backprop)
        with torch.no_grad():
            e_pred_stored = self.jepa.predictor_3.forward_with_latent(seq, g_j)
        cost_pred = self.cost_module(
            seq.detach(), e_pred_stored.detach(), g_j
        )  # [1] — graphe vivant sur cost_module

        return cost_pred, g_j, e_goal, seq

    def store_prediction(self, cost_pred, n_pellets_t, pac_xy_t, ghost_xy_t):
        """Stocke la prédiction pour backprop futur."""
        self.buffer_preds.append(
            (cost_pred, n_pellets_t, pac_xy_t, ghost_xy_t)
        )

    def maybe_backprop(
        self,
        n_pellets_now: int,
        pac_xy_now: Coord,
        ghost_xy_now: Coord,
        outcome_eaten: bool,
        optimizer: torch.optim.Optimizer,
    ) -> Optional[float]:
        """
        Si le buffer atteint j entrées, dépile la plus ancienne et backpropage.
        Retourne la loss ou None.
        """
        if len(self.buffer_preds) < self.config.j:
            return None

        cost_pred, n_pellets_t, pac_xy_t, ghost_xy_t = self.buffer_preds.popleft()

        # Coût réel
        if self.is_pacman:
            died = outcome_eaten and len(self.buffer_preds) == 0
            if died:
                real_cost = float(self.config.j * 10)
            else:
                eaten = n_pellets_t - n_pellets_now
                real_cost = float(self.config.j - eaten)
        else:
            ghost_idx = self.agent_id - 1
            gx_t, gy_t   = ghost_xy_t[ghost_idx]
            px_t, py_t   = pac_xy_t
            gx_n, gy_n   = ghost_xy_now[ghost_idx]
            px_n, py_n   = pac_xy_now
            dist_t  = abs(gx_t - px_t) + abs(gy_t - py_t)
            dist_tj = abs(gx_n - px_n) + abs(gy_n - py_n)
            real_cost = float(2 * dist_tj - dist_t)

        target = torch.tensor([real_cost], dtype=torch.float32,
                               device=self.device)
        loss   = F.mse_loss(cost_pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val = loss.item()
        self.total_loss += val
        self.n_updates   += 1
        return val

    def flush_buffer(
        self,
        history: List,              # liste de (n_pellets, pac_xy, ghost_xy)
        outcome_eaten: bool,
        optimizer: torch.optim.Optimizer,
    ):
        """
        En fin d'épisode, traite les prédictions restantes dans le buffer
        en utilisant les snapshots d'historique disponibles.
        """
        offset = 0
        while self.buffer_preds:
            cost_pred, n_pellets_t, pac_xy_t, ghost_xy_t = (
                self.buffer_preds.popleft()
            )
            # Prendre le snapshot le plus proche disponible dans l'historique
            snap_idx = min(offset + self.config.j, len(history) - 1)
            n_pellets_now, pac_xy_now, ghost_xy_now = history[snap_idx]

            if self.is_pacman:
                died = outcome_eaten and snap_idx == len(history) - 1
                real_cost = (float(self.config.j * 10) if died
                             else float(self.config.j - (n_pellets_t - n_pellets_now)))
            else:
                ghost_idx = self.agent_id - 1
                gx_t, gy_t = ghost_xy_t[ghost_idx]
                px_t, py_t = pac_xy_t
                gx_n, gy_n = ghost_xy_now[ghost_idx]
                px_n, py_n = pac_xy_now
                dist_t  = abs(gx_t - px_t) + abs(gy_t - py_t)
                dist_tj = abs(gx_n - px_n) + abs(gy_n - py_n)
                real_cost = float(2 * dist_tj - dist_t)

            target = torch.tensor([real_cost], dtype=torch.float32,
                                   device=self.device)
            loss = F.mse_loss(cost_pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.total_loss += loss.item()
            self.n_updates  += 1
            offset += 1


# ── Boucle RL principale ──────────────────────────────────────────────────────

def run_rl(
    maze_json:    str,
    jepa_weights: str,
    jepa_config:  str,
    n_episodes:   int = 200,
    lr:           float = LR_DEFAULT,
    save_every:   int = 50,
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

    # ── Charger les modules de coût (ou initialiser) ──────────────────────────
    cost_pacman = CostModule(config.d_model).to(device)
    cost_ghost  = CostModule(config.d_model).to(device)

    pac_weights_path   = os.path.join(OUTPUT_DIR, "cost_pacman.pt")
    ghost_weights_path = os.path.join(OUTPUT_DIR, "cost_ghost.pt")

    if os.path.exists(pac_weights_path):
        cost_pacman.load_state_dict(
            torch.load(pac_weights_path, map_location="cpu")
        )
        print(f"cost_pacman chargé depuis {pac_weights_path}")
    if os.path.exists(ghost_weights_path):
        cost_ghost.load_state_dict(
            torch.load(ghost_weights_path, map_location="cpu")
        )
        print(f"cost_ghost chargé depuis {ghost_weights_path}")

    optimizer = torch.optim.Adam(
        list(cost_pacman.parameters()) + list(cost_ghost.parameters()), lr=lr
    )

    # ── Préparer le jeu ───────────────────────────────────────────────────────
    maze_data = load_maze(maze_json)
    meta      = maze_data.get("metadata", {})
    width     = meta.get("width", 28)
    height    = meta.get("height", 31)

    coo_src, coo_dst, node_to_index, index_to_coord, adjacency = build_graph(
        maze_data["nodes"], maze_data["edges"], width, height
    )
    energizers_full = symmetrize_energizers(
        maze_data["energizers"], node_to_index, height
    )
    pacman_spawn, ghost_spawns = extract_walkable_and_spawns(
        maze_data["nodes"], node_to_index, maze_data.get("ghost_walls", set())
    )

    graph = dict(
        coo_src=coo_src, coo_dst=coo_dst,
        node_to_index=node_to_index, index_to_coord=index_to_coord,
        adjacency=adjacency, ghost_wall_nodes=set(),
        num_nodes=len(node_to_index),
    )
    spawns = dict(pacman_spawn=pacman_spawn, ghost_spawns=ghost_spawns)

    # ── Initialiser les agents ────────────────────────────────────────────────
    agents = [
        JEPAAgent(0, jepa, cost_pacman, config, device),   # Pac-Man
        JEPAAgent(1, jepa, cost_ghost,  config, device),   # Ghost 0
        JEPAAgent(2, jepa, cost_ghost,  config, device),   # Ghost 1
        JEPAAgent(3, jepa, cost_ghost,  config, device),   # Ghost 2
        JEPAAgent(4, jepa, cost_ghost,  config, device),   # Ghost 3
    ]

    # ── Boucle sur les épisodes ───────────────────────────────────────────────
    for ep in range(n_episodes):
        state = GameState(
            adjacency=adjacency,
            node_to_index=node_to_index,
            index_to_coord=index_to_coord,
            ghost_wall_nodes=set(),
            pacman_spawn=spawns["pacman_spawn"],
            ghost_spawns=spawns["ghost_spawns"],
            initial_pellets=set(node_to_index[tuple(p)]
                                for p in maze_data.get("pellets", set())
                                if tuple(p) in node_to_index),
            initial_energizers=set(node_to_index[e]
                                   for e in energizers_full
                                   if e in node_to_index),
        )

        coord_adj = build_coord_adj(state)
        for agent in agents:
            agent.reset()

        # Historique pour flush en fin d'épisode
        # Chaque entrée : (n_pellets, pac_xy, ghost_xy_list)
        history = []
        outcome_eaten = False

        step = 0
        done = False

        while not done:
            pac_coord   = state.index_to_coord[state.pacman_pos]
            ghost_coords = [
                state.index_to_coord[state.ghost_positions[g]]
                for g in range(4)
            ]
            active_pellets    = {state.index_to_coord[p]
                                  for p in state.active_pellets}
            active_energizers = {state.index_to_coord[e]
                                  for e in state.active_energizers}

            # Snapshot de l'état courant pour le calcul des coûts futurs
            n_pellets_now = len(state.active_pellets) + len(state.active_energizers)
            history.append((n_pellets_now, pac_coord, list(ghost_coords)))

            # ── Construire observations et choisir les actions ─────────────
            actions_chosen = {}

            for agent in agents:
                if agent.is_pacman:
                    agent_coord = pac_coord
                    gc = ghost_coords
                    pc = pac_coord
                else:
                    agent_coord = ghost_coords[agent.agent_id - 1]
                    gc = ghost_coords
                    pc = pac_coord

                cases, pos, mask = build_obs_tensors(
                    agent_coord, coord_adj,
                    active_pellets, active_energizers,
                    gc, pc, config.d_model,
                )

                # Action précédente (from last context entry or -1)
                prev_action = (
                    int(agent.context[-1][3]) if agent.context else -1
                )
                agent.observe(cases, pos, mask, prev_action)

                # Choisir action + stocker prédiction
                cost_pred, g_j, e_goal, seq = agent.choose_action()
                agent.store_prediction(
                    cost_pred, n_pellets_now, pac_coord, list(ghost_coords)
                )

                # Action niveau 1
                with torch.no_grad():
                    # Décompose e_goal en sous-objectifs de niveau 2
                    try:
                        subgoals = jepa.plan_level2(
                            seq, e_goal, n_steps=10, lr=0.05
                        )
                        e_subgoal = subgoals[0]
                    except Exception:
                        e_subgoal = e_goal

                    action_idx = jepa.choose_action(seq, e_subgoal)

                if agent.is_pacman:
                    actions_chosen["pacman"] = action_idx
                else:
                    actions_chosen[f"ghost_{agent.agent_id - 1}"] = action_idx

            # ── Backprop des prédictions arrivées à maturité (j pas) ───────
            for agent in agents:
                agent.maybe_backprop(
                    n_pellets_now, pac_coord, list(ghost_coords),
                    outcome_eaten=False, optimizer=optimizer,
                )

            # ── Pas de simulation ──────────────────────────────────────────
            done, outcome = sim_step(state, is_first_step=(step == 0))
            if outcome == "eaten":
                outcome_eaten = True
            step += 1

        # ── Fin d'épisode : traiter les prédictions restantes ──────────────
        for agent in agents:
            agent.flush_buffer(history, outcome_eaten, optimizer)

        # ── Log ───────────────────────────────────────────────────────────
        pac_loss = (agents[0].total_loss / max(agents[0].n_updates, 1))
        gh_loss  = sum(a.total_loss for a in agents[1:]) / max(
            sum(a.n_updates for a in agents[1:]), 1
        )
        print(
            f"Ep {ep+1:4d}/{n_episodes} | steps={step:4d} | "
            f"outcome={outcome} | "
            f"loss_pac={pac_loss:.4f} loss_ghost={gh_loss:.4f}"
        )

        # Remise à zéro des compteurs par épisode
        for agent in agents:
            agent.total_loss = 0.0
            agent.n_updates  = 0

        # ── Sauvegarde périodique ──────────────────────────────────────────
        if (ep + 1) % save_every == 0:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            torch.save(cost_pacman.state_dict(),
                       os.path.join(OUTPUT_DIR, "cost_pacman_rl.pt"))
            torch.save(cost_ghost.state_dict(),
                       os.path.join(OUTPUT_DIR, "cost_ghost_rl.pt"))
            print(f"  ✓ Poids RL sauvegardés (ep {ep+1})")

    print("\nRL terminé.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(cost_pacman.state_dict(),
               os.path.join(OUTPUT_DIR, "cost_pacman_rl.pt"))
    torch.save(cost_ghost.state_dict(),
               os.path.join(OUTPUT_DIR, "cost_ghost_rl.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze-json",    required=True)
    parser.add_argument("--jepa-weights", required=True)
    parser.add_argument("--jepa-config",  required=True)
    parser.add_argument("--n-episodes",   type=int,   default=200)
    parser.add_argument("--lr",           type=float, default=LR_DEFAULT)
    parser.add_argument("--save-every",   type=int,   default=50)
    args = parser.parse_args()

    run_rl(
        maze_json=args.maze_json,
        jepa_weights=args.jepa_weights,
        jepa_config=args.jepa_config,
        n_episodes=args.n_episodes,
        lr=args.lr,
        save_every=args.save_every,
    )