"""
JEPA Pac-Man agent adapter.

Wraps JEPAAgentLevel1 (src/jepacman/rl_loop_level1.py) to conform to the
Bierhoff Agent interface (reset / act).  Falls back to random moves when
PyTorch or the pre-trained weights are unavailable.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Literal, Optional, Set, Tuple

from ..game.maze import Maze
from ..game.types import Direction, Observation, PacmanAction, Pos

logger = logging.getLogger(__name__)

# Action int used by JEPA → Direction mapping (0=up 1=down 2=left 3=right)
_INT_TO_DIR = {
    0: Direction.UP,
    1: Direction.DOWN,
    2: Direction.LEFT,
    3: Direction.RIGHT,
    4: Direction.STAY,
}
_DIR_TO_INT = {v: k for k, v in _INT_TO_DIR.items()}

# Paths to pre-trained weights (relative to project root)
_WEIGHTS_ROOT = Path("optuna_results")
_JEPA_CONFIG  = _WEIGHTS_ROOT / "best_config_full.json"
_JEPA_WEIGHTS = _WEIGHTS_ROOT / "best_model.pt"
_COST_PARAMS  = _WEIGHTS_ROOT / "best_params_cost_modules.json"
_PAC_WEIGHTS  = _WEIGHTS_ROOT / "cost_pacman_l1.pt"  # prefer L1 weights


def _load_jepa(device):
    """Load HierarchicalJEPAcman from disk.  Returns (jepa, config) or raises."""
    import torch
    from ..jepacman.config import Config
    from ..jepacman.models.jepa import HierarchicalJEPAcman

    with open(_JEPA_CONFIG) as f:
        cfg_raw = json.load(f)
    valid  = {fd.name for fd in dataclasses.fields(Config)}
    config = Config(**{k: v for k, v in cfg_raw.items() if k in valid})

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

    return jepa, config


def _load_cost_module(config, device):
    """Load CostModule for Pac-Man from disk.  Returns module or raises."""
    import torch
    from ..jepacman.models.cost_module import CostModule

    n_heads, n_layers = 2, 2
    if _COST_PARAMS.exists():
        with open(_COST_PARAMS) as f:
            params = json.load(f)
        n_heads  = params.get("n_heads",  n_heads)
        n_layers = params.get("n_layers", n_layers)

    cost = CostModule(config.d_model, n_heads, n_layers).to(device)

    if _PAC_WEIGHTS.exists():
        cost.load_state_dict(torch.load(str(_PAC_WEIGHTS), map_location="cpu"))
    else:
        fallback = _WEIGHTS_ROOT / "cost_pacman.pt"
        if fallback.exists():
            cost.load_state_dict(torch.load(str(fallback), map_location="cpu"))
        else:
            logger.warning("jepa_pacman: no cost-module weights found — random init")

    return cost


def _maze_to_adj(maze: Maze) -> Dict[Pos, Set[Pos]]:
    """Convert maze.pacman_graph (frozensets) to a plain dict-of-sets."""
    return {pos: set(nbrs) for pos, nbrs in maze.pacman_graph.items()}


class JEPAPacmanAgent:
    """JEPA Level-1 Pac-Man agent wrapped for the Bierhoff benchmark framework."""

    name = "jepa_pacman"

    def __init__(self):
        self._agent   = None   # JEPAAgentLevel1 or None if torch unavailable
        self._adj: Optional[Dict[Pos, Set[Pos]]] = None
        self._config  = None
        self._prev_action_int: int = 4  # STAY until first real move

    def reset(self, maze: Maze, role: Literal["pacman", "ghosts"]) -> None:
        self._adj = _maze_to_adj(maze)
        self._prev_action_int = 4
        self._agent = None

        try:
            import torch
            from ..jepacman.rl_loop_level1 import JEPAAgentLevel1, get_device

            device     = get_device()
            jepa, cfg  = _load_jepa(device)
            cost       = _load_cost_module(cfg, device)

            self._config = cfg
            self._agent  = JEPAAgentLevel1(
                agent_id=0, jepa=jepa, cost_module=cost,
                config=cfg, device=device,
            )
            logger.info("jepa_pacman: JEPA model loaded (device=%s)", device)
        except Exception as exc:
            logger.warning("jepa_pacman: falling back to random — %s", exc)

    def act(self, obs: Observation) -> PacmanAction:
        if self._agent is None:
            # Fallback: random legal move
            maze_adj = self._adj or {}
            nbrs = list(maze_adj.get(obs.pacman_pos, ()))
            if not nbrs:
                return Direction.STAY
            nb   = random.choice(nbrs)
            dx   = nb[0] - obs.pacman_pos[0]
            dy   = nb[1] - obs.pacman_pos[1]
            return Direction.from_delta(dx, dy)

        from ..jepacman.rl_loop_level1 import build_obs_tensors

        cases, positions, pad_mask = build_obs_tensors(
            agent_coord       = obs.pacman_pos,
            adj               = self._adj,
            active_pellets    = set(obs.active_pellets),
            active_energizers = set(obs.active_energizers),
            ghost_coords      = list(obs.ghost_positions),
            pacman_coord      = obs.pacman_pos,
            d_model           = self._config.d_model,
        )

        self._agent.observe(cases, positions, pad_mask, self._prev_action_int)
        action_int, *_ = self._agent.choose_action()

        if action_int is None:
            action_int = random.randint(0, 3)

        self._prev_action_int = action_int
        return _INT_TO_DIR.get(action_int, Direction.STAY)
