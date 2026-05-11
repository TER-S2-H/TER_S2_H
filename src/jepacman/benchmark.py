"""
Benchmark des trois approches de jeu :
  - heuristic   : BFS vers le pellet le plus proche (Pac-Man) / BFS vers Pac-Man (Fantômes)
  - jepa_hier   : JEPA hiérarchique 3 niveaux (rl_loop.py)
  - jepa_l1     : JEPA niveau 1, chaîne de 5 pas (rl_loop_level1.py)

Pour chaque combinaison (pac_type × ghost_type) on joue N_EPISODES parties et on mesure :
  - durée en steps
  - taux de pellets mangés
  - taux de victoire de Pac-Man

Usage :
    python -m src.jepacman.benchmark \\
        --maze-dir      generated_mazes \\
        --jepa-weights  optuna_results/best_model.pt \\
        --jepa-config   optuna_results/best_config_full.json \\
        --n-episodes    30 \\
        --output-dir    benchmark_results
"""

import os
import sys
import json
import glob
import random
import argparse
import dataclasses
import itertools
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .config import Config
from .models.jepa import HierarchicalJEPAcman
from .models.cost_module import CostModule
from .rl_loop import (
    JEPAAgent, build_coord_adj, build_obs_tensors, plan_with_cost, get_device
)
from .rl_loop_level1 import JEPAAgentLevel1, CHAIN_LEN

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.pacman_simulator import (
    load_maze, build_graph, extract_walkable_and_spawns,
    symmetrize_energizers, GameState, step as sim_step,
)

Coord  = Tuple[int, int]
OUTPUT_DIR  = "optuna_results"
APPROACHES  = ["heuristic", "jepa_hier", "jepa_l1"]
APPROACH_LABELS = {
    "heuristic": "Heuristique",
    "jepa_hier": "JEPA Hiérar.",
    "jepa_l1":   "JEPA L1×5",
}
REVERSE_MAP = {0: 1, 1: 0, 2: 3, 3: 2}


# ── BFS helpers ───────────────────────────────────────────────────────────────

def bfs_path(start: int, targets: Set[int],
             adjacency: Dict[int, Set[int]]) -> Optional[List[int]]:
    """Retourne le chemin (liste de node_idx) de start vers le target le plus proche."""
    if start in targets:
        return [start]
    queue   = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        for nb in adjacency.get(node, set()):
            if nb not in visited:
                new_path = path + [nb]
                if nb in targets:
                    return new_path
                visited.add(nb)
                queue.append((nb, new_path))
    return None


def path_to_action(cur: int, nxt: int,
                   index_to_coord: Dict[int, Coord], width: int) -> int:
    cx, cy = index_to_coord[cur]
    nx, ny = index_to_coord[nxt]
    dx, dy = nx - cx, ny - cy
    # Tunnel wrap
    if dx >  width // 2: dx = -1
    if dx < -width // 2: dx =  1
    if dy < 0: return 0   # Nord
    if dy > 0: return 1   # Sud
    if dx < 0: return 2   # Ouest
    if dx > 0: return 3   # Est
    return random.randint(0, 3)


# ── Agents heuristiques ───────────────────────────────────────────────────────

class HeuristicPacman:
    """BFS vers le pellet (ou énergiseur) le plus proche."""

    def reset(self): pass

    def choose_action(self, state: GameState, width: int) -> int:
        targets = state.active_pellets | state.active_energizers
        if not targets:
            return random.randint(0, 3)
        path = bfs_path(state.pacman_pos, targets, state.adjacency)
        if path is None or len(path) < 2:
            return random.randint(0, 3)
        return path_to_action(path[0], path[1], state.index_to_coord, width)


class HeuristicGhost:
    """BFS vers la position de Pac-Man."""

    def reset(self): pass

    def choose_action(self, state: GameState, ghost_idx: int, width: int) -> int:
        ghost_pos = state.ghost_positions[ghost_idx]
        target    = {state.pacman_pos}
        path = bfs_path(ghost_pos, target, state.adjacency)
        if path is None or len(path) < 2:
            return random.randint(0, 3)
        return path_to_action(path[0], path[1], state.index_to_coord, width)


# ── Wrappers JEPA pour le benchmark (pas de backprop) ────────────────────────

class BenchJEPAHier:
    """Pac-Man JEPA hiérarchique — sélection d'action uniquement."""

    def __init__(self, jepa: HierarchicalJEPAcman, cost_module: CostModule,
                 config: Config, device: torch.device):
        self.agent = JEPAAgent(0, jepa, cost_module, config, device)
        self.jepa  = jepa

    def reset(self):
        self.agent.reset()

    def observe(self, cases, pos, mask, action):
        self.agent.observe(cases, pos, mask, action)

    def choose_action(self) -> int:
        if len(self.agent.context) == 0:
            return random.randint(0, 3)
        g_j, e_goal, seq, _ = self.agent.choose_action()
        with torch.no_grad():
            try:
                subgoals  = self.jepa.plan_level2(seq, e_goal, n_steps=10, lr=0.05)
                e_subgoal = subgoals[0]
            except Exception:
                e_subgoal = e_goal
            forbidden  = ({REVERSE_MAP[self.agent.last_action]}
                          if self.agent.last_action is not None else None)
            action_idx = self.jepa.choose_action(seq, e_subgoal, forbidden=forbidden)
        self.agent.last_action = action_idx
        return action_idx


class BenchJEPAHierGhost:
    """Fantôme JEPA hiérarchique."""

    def __init__(self, agent_id: int, jepa: HierarchicalJEPAcman,
                 cost_module: CostModule, config: Config, device: torch.device):
        self.agent = JEPAAgent(agent_id, jepa, cost_module, config, device)
        self.jepa  = jepa

    def reset(self):
        self.agent.reset()

    def observe(self, cases, pos, mask, action):
        self.agent.observe(cases, pos, mask, action)

    def choose_action(self) -> int:
        if len(self.agent.context) == 0:
            return random.randint(0, 3)
        g_j, e_goal, seq, _ = self.agent.choose_action()
        with torch.no_grad():
            try:
                subgoals  = self.jepa.plan_level2(seq, e_goal, n_steps=10, lr=0.05)
                e_subgoal = subgoals[0]
            except Exception:
                e_subgoal = e_goal
            forbidden  = ({REVERSE_MAP[self.agent.last_action]}
                          if self.agent.last_action is not None else None)
            action_idx = self.jepa.choose_action(seq, e_subgoal, forbidden=forbidden)
        self.agent.last_action = action_idx
        return action_idx


class BenchJEPAL1:
    """Agent JEPA L1 — sélection d'action uniquement (pas de backprop)."""

    def __init__(self, agent_id: int, jepa: HierarchicalJEPAcman,
                 cost_module: CostModule, config: Config, device: torch.device):
        self.agent = JEPAAgentLevel1(agent_id, jepa, cost_module, config, device)

    def reset(self):
        self.agent.reset()

    def observe(self, cases, pos, mask, action):
        self.agent.observe(cases, pos, mask, action)

    def choose_action(self) -> int:
        action, _, _, _ = self.agent.choose_action()
        self.agent.last_action = action
        return action


# ── Episode ───────────────────────────────────────────────────────────────────

def run_episode(
    pac_agent,
    ghost_agents,         # liste de 4 agents
    maze_json_path: str,
    config:         Config,
    device:         torch.device,
    max_steps:      int = 2000,
) -> Dict:
    maze_data = load_maze(maze_json_path)
    meta      = maze_data.get("metadata", {})
    width     = meta.get("width",  28)
    height    = meta.get("height", 31)

    _, _, node_to_index, index_to_coord, adjacency = build_graph(
        maze_data["nodes"], maze_data["edges"], width, height
    )
    energizers_full = symmetrize_energizers(
        maze_data["energizers"], node_to_index, height
    )
    pacman_spawn, ghost_spawns = extract_walkable_and_spawns(
        maze_data["nodes"], node_to_index, maze_data.get("ghost_walls", set())
    )

    state = GameState(
        adjacency          = adjacency,
        node_to_index      = node_to_index,
        index_to_coord     = index_to_coord,
        ghost_wall_nodes   = set(),
        pacman_spawn       = pacman_spawn,
        ghost_spawns       = ghost_spawns,
        initial_pellets    = set(
            node_to_index[tuple(p)]
            for p in maze_data.get("pellets", set())
            if tuple(p) in node_to_index
        ),
        initial_energizers = set(
            node_to_index[e] for e in energizers_full if e in node_to_index
        ),
    )

    initial_pellets = len(state.active_pellets) + len(state.active_energizers)
    coord_adj = build_coord_adj(state)

    # Reset agents
    pac_agent.reset()
    for g in ghost_agents:
        g.reset()

    # Heuristic flag
    pac_is_heuristic   = isinstance(pac_agent, HeuristicPacman)
    ghost_is_heuristic = isinstance(ghost_agents[0], HeuristicGhost)

    step    = 0
    done    = False
    outcome = "timeout"

    while not done and step < max_steps:
        pac_coord    = state.index_to_coord[state.pacman_pos]
        ghost_coords = [state.index_to_coord[state.ghost_positions[g]] for g in range(4)]
        active_pellets    = {state.index_to_coord[p] for p in state.active_pellets}
        active_energizers = {state.index_to_coord[e] for e in state.active_energizers}

        # ── Action Pac-Man ────────────────────────────────────────────────────
        if pac_is_heuristic:
            pac_action = pac_agent.choose_action(state, width)
        else:
            cases, pos, mask = build_obs_tensors(
                pac_coord, coord_adj, active_pellets, active_energizers,
                ghost_coords, pac_coord, config.d_model,
            )
            prev_action = int(pac_agent.agent.context[-1][3]) if pac_agent.agent.context else -1
            pac_agent.observe(cases, pos, mask, prev_action)
            pac_action = pac_agent.choose_action()

        # ── Actions Fantômes ──────────────────────────────────────────────────
        ghost_actions = []
        for i, g_agent in enumerate(ghost_agents):
            if ghost_is_heuristic:
                ghost_actions.append(g_agent.choose_action(state, i, width))
            else:
                agent_coord = ghost_coords[i]
                cases, pos, mask = build_obs_tensors(
                    agent_coord, coord_adj, active_pellets, active_energizers,
                    ghost_coords, pac_coord, config.d_model,
                )
                prev_action = int(g_agent.agent.context[-1][3]) if g_agent.agent.context else -1
                g_agent.observe(cases, pos, mask, prev_action)
                ghost_actions.append(g_agent.choose_action())

        done, outcome = sim_step(
            state,
            is_first_step = (step == 0),
            pacman_action = pac_action,
            ghost_actions = ghost_actions,
        )
        step += 1

    final_pellets = len(state.active_pellets) + len(state.active_energizers)
    pellets_eaten = initial_pellets - final_pellets
    pellet_rate   = pellets_eaten / initial_pellets if initial_pellets > 0 else 0.0
    win           = (outcome == "win") or (final_pellets == 0)

    return {"steps": step, "pellet_rate": pellet_rate, "win": win, "outcome": outcome}


# ── Benchmark principal ───────────────────────────────────────────────────────

def run_benchmark(
    maze_dir:      str,
    jepa_weights:  Optional[str],
    jepa_config:   Optional[str],
    n_episodes:    int  = 30,
    max_steps:     int  = 2000,
    output_dir:    str  = "benchmark_results",
    seed:          int  = 42,
):
    random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    device     = get_device()
    maze_files = sorted(glob.glob(os.path.join(maze_dir, "**", "maze.json"), recursive=True))
    if not maze_files:
        raise FileNotFoundError(f"Aucun maze.json dans {maze_dir}")
    print(f"{len(maze_files)} labyrinthes | {n_episodes} épisodes / combinaison | device={device}")

    # ── Charger JEPA si disponible ────────────────────────────────────────────
    jepa, config_jepa = None, None
    cost_pac_hier = cost_gho_hier = None
    cost_pac_l1   = cost_gho_l1   = None

    if jepa_weights and jepa_config:
        with open(jepa_config) as f:
            cfg_raw = json.load(f)
        valid       = {f.name for f in dataclasses.fields(Config)}
        config_jepa = Config(**{k: v for k, v in cfg_raw.items() if k in valid})

        jepa = HierarchicalJEPAcman(
            case_dim=config_jepa.case_dim, n_actions=config_jepa.n_actions,
            d_model=config_jepa.d_model, seq_len=config_jepa.seq_len,
            n_heads_state=config_jepa.n_heads_state, n_layers_state=config_jepa.n_layers_state,
            n_heads_seq=config_jepa.n_heads_seq, n_layers_seq=config_jepa.n_layers_seq,
            n_heads_pred_1=config_jepa.n_heads_pred_1, n_layers_pred_1=config_jepa.n_layers_pred_1,
            n_heads_pred_2=config_jepa.n_heads_pred_2, n_layers_pred_2=config_jepa.n_layers_pred_2,
            n_heads_pred_3=config_jepa.n_heads_pred_3, n_layers_pred_3=config_jepa.n_layers_pred_3,
            ema_momentum=config_jepa.ema_momentum, k=config_jepa.k, j=config_jepa.j,
        )
        jepa.load_state_dict(torch.load(jepa_weights, map_location="cpu"))
        jepa.to(device).eval()
        for p in jepa.parameters():
            p.requires_grad = False

        # Architecture cost modules
        def load_cost_arch(json_path_l1, json_path_base):
            p = json_path_l1 if os.path.exists(json_path_l1) else json_path_base
            if os.path.exists(p):
                with open(p) as f:
                    d = json.load(f)
                return d.get("n_heads", 2), d.get("n_layers", 2)
            return 2, 2

        nh_hier, nl_hier = load_cost_arch(
            os.path.join(OUTPUT_DIR, "best_params_cost_modules.json"),
            os.path.join(OUTPUT_DIR, "best_params_cost_modules.json"),
        )
        nh_l1, nl_l1 = load_cost_arch(
            os.path.join(OUTPUT_DIR, "best_params_cost_modules_l1.json"),
            os.path.join(OUTPUT_DIR, "best_params_cost_modules.json"),
        )

        def load_weights(module, *paths):
            for p in paths:
                if os.path.exists(p):
                    module.load_state_dict(torch.load(p, map_location="cpu"))
                    print(f"  chargé {p}")
                    return
            print(f"  ⚠ aucun poids trouvé parmi {paths}")

        cost_pac_hier = CostModule(config_jepa.d_model, nh_hier, nl_hier).to(device)
        cost_gho_hier = CostModule(config_jepa.d_model, nh_hier, nl_hier).to(device)
        cost_pac_l1   = CostModule(config_jepa.d_model, nh_l1,   nl_l1  ).to(device)
        cost_gho_l1   = CostModule(config_jepa.d_model, nh_l1,   nl_l1  ).to(device)

        load_weights(cost_pac_hier,
                     os.path.join(OUTPUT_DIR, "cost_pacman_rl.pt"),
                     os.path.join(OUTPUT_DIR, "cost_pacman.pt"))
        load_weights(cost_gho_hier,
                     os.path.join(OUTPUT_DIR, "cost_ghost_rl.pt"),
                     os.path.join(OUTPUT_DIR, "cost_ghost.pt"))
        load_weights(cost_pac_l1,
                     os.path.join(OUTPUT_DIR, "cost_pacman_rl_l1.pt"),
                     os.path.join(OUTPUT_DIR, "cost_pacman_l1.pt"))
        load_weights(cost_gho_l1,
                     os.path.join(OUTPUT_DIR, "cost_ghost_rl_l1.pt"),
                     os.path.join(OUTPUT_DIR, "cost_ghost_l1.pt"))

    # Approches disponibles
    available = ["heuristic"]
    if jepa is not None:
        available += ["jepa_hier", "jepa_l1"]

    # ── Factories d'agents ────────────────────────────────────────────────────

    def make_pac(approach):
        if approach == "heuristic":
            return HeuristicPacman()
        elif approach == "jepa_hier":
            return BenchJEPAHier(jepa, cost_pac_hier, config_jepa, device)
        else:
            return BenchJEPAL1(0, jepa, cost_pac_l1, config_jepa, device)

    def make_ghosts(approach):
        if approach == "heuristic":
            return [HeuristicGhost() for _ in range(4)]
        elif approach == "jepa_hier":
            return [BenchJEPAHierGhost(i+1, jepa, cost_gho_hier, config_jepa, device)
                    for i in range(4)]
        else:
            return [BenchJEPAL1(i+1, jepa, cost_gho_l1, config_jepa, device)
                    for i in range(4)]

    # ── Collecte des résultats ────────────────────────────────────────────────
    # results[(pac, ghost)] = list de dicts {steps, pellet_rate, win}
    results = {}

    combos = list(itertools.product(available, repeat=2))
    for combo_idx, (pac_type, ghost_type) in enumerate(combos):
        key = (pac_type, ghost_type)
        print(f"\n[{combo_idx+1}/{len(combos)}] Pac={APPROACH_LABELS[pac_type]}"
              f" | Ghosts={APPROACH_LABELS[ghost_type]}")

        pac_agent    = make_pac(pac_type)
        ghost_agents = make_ghosts(ghost_type)
        ep_results   = []

        for ep in range(n_episodes):
            maze_json_path = random.choice(maze_files)
            cfg = config_jepa if config_jepa else Config()
            res = run_episode(
                pac_agent, ghost_agents, maze_json_path,
                cfg, device, max_steps=max_steps,
            )
            ep_results.append(res)
            marker = "✓" if res["win"] else ("✗" if res["outcome"]=="eaten" else "—")
            print(f"  ep{ep+1:3d} {marker} steps={res['steps']:4d}"
                  f" pellets={res['pellet_rate']:.2%}")

        results[key] = ep_results

        # Moyennes rapides
        wins  = sum(r["win"]         for r in ep_results) / n_episodes
        prate = sum(r["pellet_rate"] for r in ep_results) / n_episodes
        steps = sum(r["steps"]       for r in ep_results) / n_episodes
        print(f"  → win={wins:.0%}  pellets={prate:.2%}  steps={steps:.0f}")

    # ── Sauvegarder JSON ──────────────────────────────────────────────────────
    save_path = os.path.join(output_dir, "results.json")
    serializable = {
        f"{k[0]}_vs_{k[1]}": v for k, v in results.items()
    }
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nRésultats sauvegardés → {save_path}")

    # ── Visualisation ─────────────────────────────────────────────────────────
    plot_results(results, available, output_dir)


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_results(results: Dict, available: List[str], output_dir: str):
    n   = len(available)
    metrics = [
        ("steps",       "Durée (steps)",          "blues"),
        ("pellet_rate", "Taux de pellets mangés",  "greens"),
        ("win",         "Taux de victoire Pac-Man","reds"),
    ]

    COLORS = {
        "heuristic": "#4C72B0",
        "jepa_hier": "#DD8452",
        "jepa_l1":   "#55A868",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Benchmark : Pac-Man vs Fantômes — 3 approches", fontsize=14, fontweight="bold")

    for ax, (metric, ylabel, _) in zip(axes, metrics):
        # Heatmap n×n
        mat   = np.zeros((n, n))
        for i, pac in enumerate(available):
            for j, ghost in enumerate(available):
                vals = [r[metric] for r in results.get((pac, ghost), [])]
                mat[i, j] = np.mean(vals) if vals else 0.0

        im = ax.imshow(mat, cmap="RdYlGn" if metric != "steps" else "RdYlGn_r",
                       vmin=0, vmax=(1 if metric != "steps" else None), aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8)

        labels = [APPROACH_LABELS[a] for a in available]
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(labels)
        ax.set_xlabel("Fantômes"); ax.set_ylabel("Pac-Man")
        ax.set_title(ylabel)

        # Valeurs dans les cellules
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                txt = f"{v:.0%}" if metric in ("pellet_rate", "win") else f"{v:.0f}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="black")

    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "heatmaps.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    print(f"Heatmaps → {heatmap_path}")
    plt.show()

    # ── Box plots par combinaison ──────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle("Distribution par combinaison Pac-Man × Fantômes", fontsize=13, fontweight="bold")

    combos = list(itertools.product(available, repeat=2))
    xlabels = [f"{APPROACH_LABELS[p][:4]}\nvs\n{APPROACH_LABELS[g][:4]}"
               for p, g in combos]

    for ax, (metric, ylabel, _) in zip(axes2, metrics):
        data = [
            [r[metric] for r in results.get(combo, [])]
            for combo in combos
        ]
        colors = [COLORS[pac] for pac, _ in combos]
        bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(combos) + 1))
        ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", alpha=0.3)

    # Légende couleurs = approche Pac-Man
    legend_handles = [mpatches.Patch(color=COLORS[a], label=APPROACH_LABELS[a])
                      for a in available]
    fig2.legend(handles=legend_handles, title="Pac-Man", loc="lower center",
                ncol=len(available), bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, "boxplots.png")
    plt.savefig(boxplot_path, dpi=150, bbox_inches="tight")
    print(f"Boxplots   → {boxplot_path}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze-dir",     default="generated_mazes")
    parser.add_argument("--jepa-weights", default=None)
    parser.add_argument("--jepa-config",  default=None)
    parser.add_argument("--n-episodes",   type=int, default=30)
    parser.add_argument("--max-steps",    type=int, default=2000)
    parser.add_argument("--output-dir",   default="benchmark_results")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    run_benchmark(
        maze_dir     = args.maze_dir,
        jepa_weights = args.jepa_weights,
        jepa_config  = args.jepa_config,
        n_episodes   = args.n_episodes,
        max_steps    = args.max_steps,
        output_dir   = args.output_dir,
        seed         = args.seed,
    )
