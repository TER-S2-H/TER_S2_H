"""
JEPA agent adapters for the Bierhoff benchmark framework.

Four agents are exposed:
  jepa_hier_pacman  — HierarchicalJEPAcman + predictor_3 goal optimisation
  jepa_l1_pacman    — JEPAAgentLevel1 (chain of 5 predictions)
  jepa_hier_ghosts  — 4 × hierarchical agents, one per ghost
  jepa_l1_ghosts    — 4 × L1 agents, one per ghost

All agents fall back to random legal moves when PyTorch is unavailable or
the pre-trained weights cannot be loaded.

Action convention (shared with pacman_simulator.py):
  0 = UP  1 = DOWN  2 = LEFT  3 = RIGHT  4 = STAY
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

from ..game.maze import Maze
from ..game.types import Direction, GhostAction, Observation, PacmanAction, Pos

logger = logging.getLogger(__name__)

_INT_TO_DIR = {
    0: Direction.UP,
    1: Direction.DOWN,
    2: Direction.LEFT,
    3: Direction.RIGHT,
    4: Direction.STAY,
}

_WEIGHTS = Path("optuna_results")
_JEPA_CONFIG  = _WEIGHTS / "best_config_full.json"
_JEPA_WEIGHTS = _WEIGHTS / "best_model.pt"
_COST_PARAMS  = _WEIGHTS / "best_params_cost_modules.json"


# ---------------------------------------------------------------------------
# Shared model loading helpers
# ---------------------------------------------------------------------------

def _load_config():
    import dataclasses as dc
    from ..jepacman.config import Config
    with open(_JEPA_CONFIG) as f:
        raw = json.load(f)
    valid = {fd.name for fd in dc.fields(Config)}
    return Config(**{k: v for k, v in raw.items() if k in valid})


def _load_jepa(config, device):
    import torch
    from ..jepacman.models.jepa import HierarchicalJEPAcman
    jepa = HierarchicalJEPAcman(
        case_dim=config.case_dim, n_actions=config.n_actions,
        d_model=config.d_model,   seq_len=config.seq_len,
        n_heads_state=config.n_heads_state, n_layers_state=config.n_layers_state,
        n_heads_seq=config.n_heads_seq,     n_layers_seq=config.n_layers_seq,
        n_heads_pred_1=config.n_heads_pred_1, n_layers_pred_1=config.n_layers_pred_1,
        n_heads_pred_2=config.n_heads_pred_2, n_layers_pred_2=config.n_layers_pred_2,
        n_heads_pred_3=config.n_heads_pred_3, n_layers_pred_3=config.n_layers_pred_3,
        ema_momentum=config.ema_momentum, k=config.k, j=config.j,
    )
    jepa.load_state_dict(torch.load(str(_JEPA_WEIGHTS), map_location="cpu"))
    jepa.to(device).eval()
    for p in jepa.parameters():
        p.requires_grad = False
    return jepa


def _load_cost(config, device, is_pacman: bool, l1: bool):
    import torch
    from ..jepacman.models.cost_module import CostModule
    n_heads, n_layers = 2, 2
    if _COST_PARAMS.exists():
        with open(_COST_PARAMS) as f:
            params = json.load(f)
        n_heads  = params.get("n_heads",  n_heads)
        n_layers = params.get("n_layers", n_layers)
    cost = CostModule(config.d_model, n_heads, n_layers).to(device)
    prefix = "pacman" if is_pacman else "ghost"
    suffix = "_l1" if l1 else ""
    w = _WEIGHTS / f"cost_{prefix}{suffix}.pt"
    if not w.exists():
        w = _WEIGHTS / f"cost_{prefix}.pt"   # fallback
    if w.exists():
        cost.load_state_dict(torch.load(str(w), map_location="cpu"))
    else:
        logger.warning("jepa_agents: no weights for cost_%s%s — random init", prefix, suffix)
    return cost


def _maze_adj(maze: Maze) -> Dict[Pos, Set[Pos]]:
    return {pos: set(nbrs) for pos, nbrs in maze.pacman_graph.items()}


def _random_legal(pos: Pos, adj: Dict) -> Direction:
    nbrs = list(adj.get(pos, ()))
    if not nbrs:
        return Direction.STAY
    nb = random.choice(nbrs)
    return Direction.from_delta(nb[0] - pos[0], nb[1] - pos[1])


def _obs_tensors(agent_pos, obs: Observation, adj):
    from ..jepacman.rl_loop_level1 import build_obs_tensors
    return build_obs_tensors(
        agent_coord       = agent_pos,
        adj               = adj,
        active_pellets    = set(obs.active_pellets),
        active_energizers = set(obs.active_energizers),
        ghost_coords      = list(obs.ghost_positions),
        pacman_coord      = obs.pacman_pos,
        d_model           = 1,   # unused by build_obs_tensors
    )


# ---------------------------------------------------------------------------
# Hierarchical JEPA — action decoded by nearest 1-step predictor output
# ---------------------------------------------------------------------------

def _hier_choose_action(agent, config, device) -> int:
    """
    Decode the goal latent into a first action by picking the direction
    whose predictor_1 output is closest to e_goal.
    """
    import torch
    from ..jepacman.rl_loop import plan_with_cost

    if len(agent.context) == 0:
        return random.randint(0, 3)

    seq = agent._build_seq_tensor()         # [1, T, d]
    _, e_goal = plan_with_cost(
        agent.jepa, agent.cost_module, seq,
        n_steps=20, lr=0.05,
    )                                        # e_goal: [1, d]

    with torch.no_grad():
        acts = torch.arange(4, device=device)
        e_nexts = torch.stack([
            agent.jepa.predictor_1(seq, acts[a:a+1])
            for a in range(4)
        ], dim=0)                            # [4, 1, d]
        dists = torch.norm(e_nexts - e_goal.unsqueeze(0), dim=-1).squeeze()
        return int(dists.argmin().item())


class JEPAHierPacmanAgent:
    name = "jepa_hier_pacman"

    def __init__(self):
        self._agent  = None
        self._adj    = None
        self._config = None
        self._device = None
        self._prev   = 4

    def reset(self, maze: Maze, role: Literal["pacman", "ghosts"]) -> None:
        self._adj  = _maze_adj(maze)
        self._prev = 4
        self._agent = None
        try:
            import torch
            from ..jepacman.rl_loop import JEPAAgent
            from ..jepacman.rl_loop_level1 import get_device
            device = get_device()
            cfg    = _load_config()
            jepa   = _load_jepa(cfg, device)
            cost   = _load_cost(cfg, device, is_pacman=True, l1=False)
            self._config = cfg
            self._device = device
            self._agent  = JEPAAgent(agent_id=0, jepa=jepa, cost_module=cost,
                                     config=cfg, device=device)
        except Exception as exc:
            logger.warning("jepa_hier_pacman: fallback to random — %s", exc)

    def act(self, obs: Observation) -> PacmanAction:
        if self._agent is None:
            return _random_legal(obs.pacman_pos, self._adj)
        cases, pos, mask = _obs_tensors(obs.pacman_pos, obs, self._adj)
        self._agent.observe(cases, pos, mask, self._prev)
        action = _hier_choose_action(self._agent, self._config, self._device)
        self._prev = action
        return _INT_TO_DIR.get(action, Direction.STAY)


# ---------------------------------------------------------------------------
# L1 JEPA Pac-Man
# ---------------------------------------------------------------------------

class JEPAL1PacmanAgent:
    name = "jepa_l1_pacman"

    def __init__(self):
        self._agent  = None
        self._adj    = None
        self._config = None
        self._prev   = 4

    def reset(self, maze: Maze, role: Literal["pacman", "ghosts"]) -> None:
        self._adj  = _maze_adj(maze)
        self._prev = 4
        self._agent = None
        try:
            import torch
            from ..jepacman.rl_loop_level1 import JEPAAgentLevel1, get_device
            device = get_device()
            cfg    = _load_config()
            jepa   = _load_jepa(cfg, device)
            cost   = _load_cost(cfg, device, is_pacman=True, l1=True)
            self._config = cfg
            self._agent  = JEPAAgentLevel1(agent_id=0, jepa=jepa, cost_module=cost,
                                           config=cfg, device=device)
        except Exception as exc:
            logger.warning("jepa_l1_pacman: fallback to random — %s", exc)

    def act(self, obs: Observation) -> PacmanAction:
        if self._agent is None:
            return _random_legal(obs.pacman_pos, self._adj)
        cases, pos, mask = _obs_tensors(obs.pacman_pos, obs, self._adj)
        self._agent.observe(cases, pos, mask, self._prev)
        action, *_ = self._agent.choose_action()
        if action is None:
            action = random.randint(0, 3)
        self._prev = action
        return _INT_TO_DIR.get(action, Direction.STAY)


# ---------------------------------------------------------------------------
# Ghost team wrappers (one JEPA agent per ghost)
# ---------------------------------------------------------------------------

class _JEPAGhostTeam:
    """Base for ghost-team JEPA agents."""

    def __init__(self, l1: bool):
        self._l1      = l1
        self._agents: List = []
        self._adj    = None
        self._config = None
        self._device = None
        self._prevs  = [4, 4, 4, 4]

    def reset(self, maze: Maze, role: Literal["pacman", "ghosts"]) -> None:
        self._adj    = _maze_adj(maze)
        self._prevs  = [4, 4, 4, 4]
        self._agents = []
        try:
            import torch
            from ..jepacman.rl_loop_level1 import get_device
            device = get_device()
            cfg    = _load_config()
            jepa   = _load_jepa(cfg, device)
            cost   = _load_cost(cfg, device, is_pacman=False, l1=self._l1)
            self._config = cfg
            self._device = device
            if self._l1:
                from ..jepacman.rl_loop_level1 import JEPAAgentLevel1
                for i in range(1, 5):
                    self._agents.append(
                        JEPAAgentLevel1(agent_id=i, jepa=jepa, cost_module=cost,
                                        config=cfg, device=device))
            else:
                from ..jepacman.rl_loop import JEPAAgent
                for i in range(1, 5):
                    self._agents.append(
                        JEPAAgent(agent_id=i, jepa=jepa, cost_module=cost,
                                  config=cfg, device=device))
        except Exception as exc:
            logger.warning("%s: fallback to random — %s", self.name, exc)

    def act(self, obs: Observation) -> GhostAction:
        dirs: List[Direction] = []
        for i in range(4):
            gpos = obs.ghost_positions[i]
            if not self._agents:
                dirs.append(_random_legal(gpos, self._adj))
                continue
            agent = self._agents[i]
            cases, pos, mask = _obs_tensors(gpos, obs, self._adj)
            agent.observe(cases, pos, mask, self._prevs[i])
            if self._l1:
                action, *_ = agent.choose_action()
                if action is None:
                    action = random.randint(0, 3)
            else:
                action = _hier_choose_action(agent, self._config, self._device)
            self._prevs[i] = action
            dirs.append(_INT_TO_DIR.get(action, Direction.STAY))
        return tuple(dirs)  # type: ignore


class JEPAHierGhostsAgent(_JEPAGhostTeam):
    name = "jepa_hier_ghosts"
    def __init__(self): super().__init__(l1=False)


class JEPAL1GhostsAgent(_JEPAGhostTeam):
    name = "jepa_l1_ghosts"
    def __init__(self): super().__init__(l1=True)
