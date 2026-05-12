"""
MCTS Ghost-Team agent.

Joint-action MCTS: the tree searches over the Cartesian product of all four
ghosts' legal moves.  Tactics: Trap (maximise capture × pellet-denial) and
Edible-survival (when Pac-Man holds an energizer).

Each ghost commits to a direction until it reaches a junction (≥3 neighbours),
preventing oscillation between ticks.
"""

from __future__ import annotations

import itertools
import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from ..game.maze import Maze
from ..game.simulator import SimState, step as sim_step
from ..game.types import Direction, GhostAction, Observation, Pos

logger = logging.getLogger(__name__)

_CARDINALS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GhostMCTSConfig:
    n_iterations:       int   = 40      # iterations per decision
    max_depth:          int   = 5       # tree depth (joint-action steps)
    T_time:             int   = 30      # rollout horizon
    T_visit:            int   = 2       # visits before UCT kicks in
    C:                  float = 1.5     # UCT exploration constant
    frightened_duration: int  = 40


# ---------------------------------------------------------------------------
# MCTS tree node
# ---------------------------------------------------------------------------

class GhostMCTSNode:
    """Joint-action node: one node per (parent, joint_action) pair."""

    __slots__ = (
        "joint_action",
        "depth",
        "parent",
        "children",   # joint_action -> GhostMCTSNode
        "n",
        "s_capture",
        "s_pill_denied",
        "s_ghost_safety",
    )

    def __init__(
        self,
        joint_action: Optional[GhostAction],
        depth: int,
        parent: Optional["GhostMCTSNode"] = None,
    ):
        self.joint_action = joint_action
        self.depth  = depth
        self.parent = parent
        self.children: Dict[GhostAction, "GhostMCTSNode"] = {}
        self.n             = 0
        self.s_capture     = 0.0
        self.s_pill_denied = 0.0
        self.s_ghost_safety = 0.0

    def v_i(self, tactic: str) -> float:
        if tactic == "edible_survival":
            return self.s_ghost_safety * (0.5 + 0.5 * self.s_pill_denied)
        # trap (default)
        return self.s_capture * self.s_pill_denied

    def uct_score(self, n_parent: int, C: float, tactic: str) -> float:
        if self.n == 0:
            return float("inf")
        return self.v_i(tactic) + C * math.sqrt(math.log(n_parent) / self.n)

    def update(self, r_capture: float, r_pill: float, r_safety: float):
        self.n += 1
        n = self.n
        self.s_capture     += (r_capture - self.s_capture)     / n
        self.s_pill_denied += (r_pill    - self.s_pill_denied) / n
        self.s_ghost_safety += (r_safety - self.s_ghost_safety) / n


# ---------------------------------------------------------------------------
# Legal joint-action enumeration
# ---------------------------------------------------------------------------

def _legal_joint_actions(state: SimState, maze: Maze) -> List[GhostAction]:
    """Return all legal joint actions, filtering same-edge same-direction pairs."""
    per_ghost: List[List[Direction]] = []
    for gpos in state.ghost_positions:
        legal = [
            d for d in _CARDINALS
            if d.apply(gpos) in maze.ghost_graph.get(gpos, ())
        ]
        if not legal:
            legal = [Direction.STAY]
        per_ghost.append(legal)

    result: List[GhostAction] = []
    for combo in itertools.product(*per_ghost):
        # Filter: no two ghosts move in same direction on same edge
        edges = [(state.ghost_positions[i], combo[i]) for i in range(4)]
        if len(set(edges)) == 4:
            result.append(combo)  # type: ignore

    return result if result else [tuple(Direction.STAY for _ in range(4))]  # type: ignore


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def _ghost_rollout(
    state: SimState,
    maze: Maze,
    cfg: GhostMCTSConfig,
    pellets_at_root: int,
) -> Tuple[float, float, float]:
    """
    Random rollout from state for up to T_time steps.
    Returns (r_capture, r_pill_denied, r_ghost_safety).
    """
    s = state.copy()

    for _ in range(cfg.T_time):
        if s.is_terminal:
            break

        # Pac-Man: random no-reverse
        legal_pac = [
            d for d in _CARDINALS
            if d.apply(s.pacman_pos) in maze.pacman_graph.get(s.pacman_pos, ())
        ]
        pac_action = random.choice(legal_pac) if legal_pac else Direction.STAY

        # Ghosts: random legal joint action
        joint = random.choice(_legal_joint_actions(s, maze))

        s = sim_step(s, maze, pac_action, joint,
                     frightened_duration=cfg.frightened_duration,
                     max_steps=99999)

    r_capture = 1.0 if s.outcome == "eaten" else 0.0
    remaining = len(s.active_pellets) + len(s.active_energizers)
    r_pill = 1.0 - (remaining / max(pellets_at_root, 1))
    r_safety = 1.0 if all(f == 0 for f in s.ghost_frightened) else 0.0

    return r_capture, r_pill, r_safety


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class MCTSGhostsAgent:
    """MCTS ghost-team agent (joint-action tree) with per-ghost direction commitment."""

    name = "mcts_ghosts"

    def __init__(self, cfg: Optional[GhostMCTSConfig] = None):
        self.cfg = cfg or GhostMCTSConfig()

    def reset(self, maze: Maze, role: Literal["pacman", "ghosts"]) -> None:
        self._maze = maze
        self._root: Optional[GhostMCTSNode] = None
        # Per-ghost committed direction; None means "needs a decision"
        self._committed_dirs: List[Optional[Direction]] = [None] * 4

    def _is_junction(self, gpos: Pos) -> bool:
        """A position is a junction when it has ≥ 3 neighbours (ghost must choose a branch)."""
        return len(self._maze.ghost_graph.get(gpos, ())) >= 3

    def act(self, obs: Observation) -> GhostAction:
        maze = self._maze
        cfg  = self.cfg

        state = SimState(
            pacman_pos        = obs.pacman_pos,
            ghost_positions   = list(obs.ghost_positions),
            ghost_frightened  = list(obs.ghost_frightened),
            active_pellets    = frozenset(obs.active_pellets),
            active_energizers = frozenset(obs.active_energizers),
            step              = obs.step,
        )

        tactic = "edible_survival" if any(f > 0 for f in obs.ghost_frightened) else "trap"

        # Each ghost only picks a new direction when it reaches a junction
        at_junction = [
            self._committed_dirs[i] is None or self._is_junction(obs.ghost_positions[i])
            for i in range(4)
        ]

        if any(at_junction):
            self._root = GhostMCTSNode(joint_action=None, depth=0)
            pellets_at_root = len(state.active_pellets) + len(state.active_energizers)

            for _ in range(cfg.n_iterations):
                self._run_iteration(self._root, state, maze, cfg, pellets_at_root, tactic)

            if self._root.children:
                best_action = max(
                    self._root.children.keys(),
                    key=lambda a: self._root.children[a].v_i(tactic),
                )
            else:
                best_action = random.choice(_legal_joint_actions(state, maze))

            for i in range(4):
                if at_junction[i]:
                    self._committed_dirs[i] = best_action[i]

        # Execute each ghost's committed direction if still legal; else fallback
        joint: List[Direction] = []
        for i in range(4):
            gpos = obs.ghost_positions[i]
            nbrs = maze.ghost_graph.get(gpos, ())
            cd = self._committed_dirs[i]
            if cd is not None and cd.apply(gpos) in nbrs:
                joint.append(cd)
            else:
                legal = [d for d in _CARDINALS if d.apply(gpos) in nbrs]
                chosen = random.choice(legal) if legal else Direction.STAY
                self._committed_dirs[i] = chosen
                joint.append(chosen)

        return tuple(joint)  # type: ignore

    def _run_iteration(
        self,
        root: GhostMCTSNode,
        root_state: SimState,
        maze: Maze,
        cfg: GhostMCTSConfig,
        pellets_at_root: int,
        tactic: str,
    ):
        node  = root
        state = root_state.copy()
        path: List[GhostMCTSNode] = [root]

        while True:
            if state.is_terminal or node.depth >= cfg.max_depth:
                break

            # Expand
            legal = _legal_joint_actions(state, maze)
            for ja in legal:
                if ja not in node.children:
                    node.children[ja] = GhostMCTSNode(ja, node.depth + 1, node)

            # Select
            unvisited = [n for n in node.children.values() if n.n < cfg.T_visit]
            if unvisited:
                child = random.choice(unvisited)
            else:
                child = max(
                    node.children.values(),
                    key=lambda c: c.uct_score(node.n, cfg.C, tactic),
                )

            # Simulate Pac-Man random step alongside ghost joint action
            legal_pac = [
                d for d in _CARDINALS
                if d.apply(state.pacman_pos) in maze.pacman_graph.get(state.pacman_pos, ())
            ]
            pac_action = random.choice(legal_pac) if legal_pac else Direction.STAY

            state = sim_step(state, maze, pac_action, child.joint_action,
                             frightened_duration=cfg.frightened_duration,
                             max_steps=99999)
            node = child
            path.append(node)

        # Rollout
        r_c, r_p, r_s = _ghost_rollout(state, maze, cfg, pellets_at_root)

        # Backprop (mean)
        for n in reversed(path):
            n.update(r_c, r_p, r_s)
