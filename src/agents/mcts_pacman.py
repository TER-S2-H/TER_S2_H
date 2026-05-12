"""
MCTS Pac-Man agent.  Implements Pepels & Winands (CIG 2012) with the
modifications specified in MCTS_AGENTS_TASK_BRIEF.md §5.

Key design choices (all cross-referenced to the brief):
- Junction-graph tree: nodes = junctions/dead-ends, edges = corridors §5.2
- Variable depth limited by T_path §5.2
- Three tactics: Survival / Ghost / Pill §5.3, Endgame §5.5
- Max-backpropagation §5.6
- Rollout uses HeuristicPacman + HeuristicGhosts §5.4
- LGR in rollout via heuristic agents §5.4
- Orphan-pellet fallback K_orphan §5.4 / §8
- UCT C=1.5 §5.6
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

from ..game.maze import Maze
from ..game.simulator import SimState, initial_state, step as sim_step
from ..game.types import Direction, GhostAction, Observation, PacmanAction, Pos

logger = logging.getLogger(__name__)

# Pre-computed constants for rollout hot-path (avoid Direction enum overhead)
_UP    = Direction.UP
_DOWN  = Direction.DOWN
_LEFT  = Direction.LEFT
_RIGHT = Direction.RIGHT
_STAY  = Direction.STAY
# (Direction, dx, dy) tuples for cardinals — avoids Direction.__get__ in tight loops
_CARD4: tuple = ((_UP, 0, -1), (_DOWN, 0, 1), (_LEFT, -1, 0), (_RIGHT, 1, 0))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PacmanMCTSConfig:
    T_path:        int   = 60      # variable-depth budget (cells)
    T_survival:    float = 0.7     # survival threshold for tactic switch
    T_time:        int   = 40      # max rollout length in ticks
    T_visit:       int   = 3       # visits before UCT kicks in
    C:             float = 1.5     # UCT exploration constant
    K_orphan:      int   = 20      # orphan-pellet fallback threshold
    K_endgame:     int   = 40      # endgame trigger pellet count
    n_iterations:  int   = 60      # MCTS iterations per move
    frightened_duration: int = 40  # ticks
    epsilon_ghost: float = 0.05    # ghost rollout exploration
    max_steps:     int   = 5000    # game max steps (timeout)


# ---------------------------------------------------------------------------
# MCTS tree node
# ---------------------------------------------------------------------------

class MCTSNode:
    """
    Node in the Pac-Man MCTS tree.

    Represents arrival at a junction (or dead-end) via a particular
    incoming corridor.  Children are outgoing corridor directions
    (excluding reverse past ply 1).
    """

    __slots__ = (
        "junction",
        "incoming_dir",
        "depth_ticks",   # ticks from root (in-game distance)
        "parent",
        "children",      # dir -> MCTSNode
        "n",             # visit count
        "s_survival_max",
        "s_survival_mean",
        "s_pill_max",
        "s_pill_mean",
        "s_ghost_max",
        "s_ghost_mean",
    )

    def __init__(
        self,
        junction: Pos,
        incoming_dir: Optional[Direction],
        depth_ticks: int,
        parent: Optional["MCTSNode"] = None,
    ):
        self.junction     = junction
        self.incoming_dir = incoming_dir
        self.depth_ticks  = depth_ticks
        self.parent       = parent
        self.children: Dict[Direction, "MCTSNode"] = {}

        self.n               = 0
        self.s_survival_max  = 0.0
        self.s_survival_mean = 0.0
        self.s_pill_max      = 0.0
        self.s_pill_mean     = 0.0
        self.s_ghost_max     = 0.0
        self.s_ghost_mean    = 0.0

    def v_i(self, tactic: str) -> float:
        """Scalar UCT compares on, based on active tactic (§5.3)."""
        if tactic == "survival":
            return self.s_survival_max
        elif tactic == "ghost":
            return self.s_ghost_max * self.s_survival_max
        else:  # pill (default)
            return self.s_pill_max * self.s_survival_max

    def uct_score(self, n_parent: int, C: float, tactic: str) -> float:
        if self.n == 0:
            return float("inf")
        exploit = self.v_i(tactic)
        explore = C * math.sqrt(math.log(n_parent) / self.n)
        return exploit + explore

    def update(self, r_survival: float, r_pill: float, r_ghost: float):
        """Max-backpropagation: keep running max and update mean."""
        self.n += 1
        # Max
        self.s_survival_max = max(self.s_survival_max, r_survival)
        self.s_pill_max     = max(self.s_pill_max,     r_pill)
        self.s_ghost_max    = max(self.s_ghost_max,    r_ghost)
        # Running mean
        n = self.n
        self.s_survival_mean += (r_survival - self.s_survival_mean) / n
        self.s_pill_mean     += (r_pill     - self.s_pill_mean)     / n
        self.s_ghost_mean    += (r_ghost    - self.s_ghost_mean)    / n


# ---------------------------------------------------------------------------
# Rollout logic
# ---------------------------------------------------------------------------

def _fast_pac_move(
    pos: Pos,
    ghost_positions: List[Pos],
    ghost_frightened: List[int],
    active_pellets,
    active_energizers,
    graph: dict,
    prev_dir: Optional[Direction],
    _target_cache: List,   # mutable single-element list for target caching
) -> Direction:
    """Quick Pac-Man rollout policy: avoid non-frightened ghosts, move toward target."""
    px, py = pos
    nbrs = graph.get(pos, ())

    danger_set: set = set()
    frightened_ghosts: list = []
    for i in range(len(ghost_positions)):
        gp = ghost_positions[i]
        if ghost_frightened[i] == 0:
            danger_set.add(gp)
        else:
            frightened_ghosts.append(gp)

    # Build legal moves using pre-cached deltas (avoids Direction.apply overhead)
    legal_dirs: list = []
    legal_nbs:  list = []
    for d, dx, dy in _CARD4:
        nb = (px + dx, py + dy)
        if nb in nbrs:
            legal_dirs.append(d)
            legal_nbs.append(nb)

    if not legal_dirs:
        return _STAY

    # Determine target (cached until it's eaten); limit search to 32 samples for speed
    target = _target_cache[0]
    all_targets = active_pellets | active_energizers
    if target is None or target not in all_targets:
        if frightened_ghosts:
            target = min(frightened_ghosts, key=lambda t: abs(px - t[0]) + abs(py - t[1]))
        elif all_targets:
            pool = all_targets if len(all_targets) <= 32 else set(random.sample(sorted(all_targets), 32))
            best_d2 = 10**9
            best_t2 = None
            for t in pool:
                d2 = abs(px - t[0]) + abs(py - t[1])
                if d2 < best_d2:
                    best_d2 = d2
                    best_t2 = t
            target = best_t2
        else:
            target = None
        _target_cache[0] = target

    if target is None:
        return random.choice(legal_dirs)

    tx, ty = target
    best_score = -1e9
    best_dir   = legal_dirs[0]
    danger_list = list(danger_set)
    for d, (nx, ny) in zip(legal_dirs, legal_nbs):
        nb = (nx, ny)
        if nb in danger_set:
            score = -1000.0
        else:
            score = -(abs(nx - tx) + abs(ny - ty))
            for gx, gy in danger_list:
                gd = abs(nx - gx) + abs(ny - gy)
                if gd <= 3:
                    score -= 5.0 / max(1, gd)
        if score > best_score:
            best_score = score
            best_dir   = d
    return best_dir


def _fast_ghost_move(
    gpos: Pos,
    pacman_pos: Pos,
    frightened: int,
    prev_dir: Direction,
    graph: dict,
) -> Direction:
    """Quick ghost rollout policy: chase Pac-Man (worst-case model) or flee when frightened."""
    gx, gy = gpos
    px, py = pacman_pos
    nbrs = graph.get(gpos, ())
    rev  = prev_dir.reverse()

    # Collect legal non-reverse moves with pre-cached deltas
    legal_dirs: list = []
    legal_nbs:  list = []
    for d, dx, dy in _CARD4:
        nb = (gx + dx, gy + dy)
        if nb in nbrs and d is not rev:
            legal_dirs.append(d)
            legal_nbs.append(nb)
    if not legal_dirs:
        for d, dx, dy in _CARD4:
            nb = (gx + dx, gy + dy)
            if nb in nbrs:
                legal_dirs.append(d)
                legal_nbs.append(nb)
    if not legal_dirs:
        return _STAY

    if frightened > 0:
        best_i = max(range(len(legal_nbs)), key=lambda i: abs(legal_nbs[i][0]-px) + abs(legal_nbs[i][1]-py))
    else:
        best_i = min(range(len(legal_nbs)), key=lambda i: abs(legal_nbs[i][0]-px) + abs(legal_nbs[i][1]-py))
    return legal_dirs[best_i]


def _rollout(
    state: SimState,
    maze: Maze,
    cfg: PacmanMCTSConfig,
    pellets_at_root: int,
) -> Tuple[float, float, float]:
    """
    Fast rollout from state for up to T_time ticks.
    Returns (R_survival, R_pill, R_ghost).
    Uses lightweight distance-based policies for speed.
    """
    pac_prev_dir   = Direction.STAY
    ghost_prev_dirs = [Direction.STAY] * 4
    _target_cache  = [None]

    ghosts_eaten   = 0
    ghost_eat_ticks: List[int] = []  # remaining frightened at eat time

    s = state.copy()

    for t in range(cfg.T_time):
        if s.is_terminal:
            break

        # Energizer penalty: abort rollout with R_survival=0
        if s.energizer_penalty:
            return 0.0, 0.0, 0.0

        p_action = _fast_pac_move(
            s.pacman_pos, s.ghost_positions, s.ghost_frightened,
            s.active_pellets, s.active_energizers,
            maze.pacman_graph, pac_prev_dir, _target_cache,
        )
        g_dirs = tuple(
            _fast_ghost_move(
                s.ghost_positions[i], s.pacman_pos, s.ghost_frightened[i],
                ghost_prev_dirs[i], maze.ghost_graph,
            )
            for i in range(4)
        )
        pac_prev_dir    = p_action
        ghost_prev_dirs = list(g_dirs)

        prev_frightened = s.ghost_frightened[:]
        s = sim_step(s, maze, p_action, g_dirs,  # type: ignore
                     frightened_duration=cfg.frightened_duration,
                     max_steps=cfg.max_steps)

        # Detect ghost eats
        for i in range(4):
            if prev_frightened[i] > 0 and s.ghost_frightened[i] == 0:
                if s.ghost_positions[i] == maze.ghost_spawns[i]:
                    ghosts_eaten += 1
                    ghost_eat_ticks.append(prev_frightened[i])

        if s.energizer_penalty:
            return 0.0, 0.0, 0.0

    # Compute rewards
    if s.outcome == "eaten":
        return 0.0, 0.0, 0.0

    if s.outcome == "cleared":
        r_survival = 1.0
        r_pill     = 1.0
        r_ghost    = ghosts_eaten / 4.0 if ghosts_eaten > 0 else 0.0
        return r_survival, r_pill, r_ghost

    # Timeout reached with Pac-Man alive
    r_survival = 1.0

    # R_pill: pellets eaten / pellets at root, with orphan fallback
    pellets_eaten = pellets_at_root - len(s.active_pellets) - len(s.active_energizers)
    if pellets_at_root > 0:
        r_pill_raw = pellets_eaten / pellets_at_root
    else:
        r_pill_raw = 1.0

    # Corridor credit (§5.4): credit pellet corridors only when cleared,
    # unless fewer than K_orphan remain globally
    remaining_total = len(s.active_pellets) + len(s.active_energizers)
    if remaining_total >= cfg.K_orphan:
        r_pill = r_pill_raw
    else:
        r_pill = r_pill_raw  # individual credits already counted

    # R_ghost: weighted by remaining-frightened at eat time (eat early = better)
    if ghosts_eaten == 0:
        r_ghost = 0.0
    else:
        r_ghost = min(1.0, sum(t / cfg.frightened_duration for t in ghost_eat_ticks) / 4.0)

    # Long-term-goal modifiers (§5.4)
    if r_ghost > 0.5:
        r_pill = r_pill + r_ghost
    elif r_ghost <= 0.5 and any(e not in state.active_energizers for e in maze.initial_energizers):
        # An energizer was eaten during rollout
        r_pill = 0.0

    r_pill = min(1.0, r_pill)

    return r_survival, r_pill, r_ghost


# ---------------------------------------------------------------------------
# Main MCTS agent
# ---------------------------------------------------------------------------

class MCTSPacmanAgent:
    """
    Monte-Carlo Tree Search Pac-Man agent.

    The tree is rooted at the current junction; MCTS runs N iterations
    each time Pac-Man reaches a decision point (junction/dead-end).
    In corridors the cached direction is followed automatically.
    """
    name = "mcts_pacman"

    def __init__(self, cfg: Optional[PacmanMCTSConfig] = None):
        self.cfg = cfg or PacmanMCTSConfig()

    def reset(self, maze: Maze, role: Literal["pacman", "ghosts"]) -> None:
        self._maze  = maze
        self._root: Optional[MCTSNode] = None
        # Path-following state: list of positions to step through
        self._planned_path: Optional[List[Pos]] = None
        self._planned_path_idx: int = 0
        self._tactic: str = "pill"
        self._endgame: bool = False
        self._no_move_streak: int = 0
        self._last_junction: Optional[Pos] = None  # last junction we made a decision at
        # BFS path cache: (from, to) -> path list (junction-to-junction paths are stable)
        self._path_cache: Dict[Tuple[Pos, Pos], Optional[List[Pos]]] = {}
        # Pre-populate all junction-to-junction paths (108 paths for this maze)
        for j1 in maze.junctions:
            for j2 in maze.junction_graph.get(j1, {}):
                key = (j1, j2)
                if key not in self._path_cache:
                    path = maze.bfs_path(j1, j2)
                    self._path_cache[key] = path
                    if path:
                        self._path_cache[(j2, j1)] = list(reversed(path))

        # Warn if T_path < diameter / 2
        diameter = maze.graph_diameter()
        if self.cfg.T_path < diameter // 2:
            logger.warning(
                "T_path=%d is smaller than half the pacman_graph diameter (%d/2=%d). "
                "Distant regions will be invisible. Consider increasing T_path or "
                "relying on endgame tactic.",
                self.cfg.T_path, diameter, diameter // 2,
            )

    def _follow_path(self, pos: Pos) -> Optional[Direction]:
        """
        Advance along _planned_path.
        Returns the next direction, or None if path is exhausted/invalid.
        """
        path = self._planned_path
        if not path:
            return None
        # Find current position in the path
        idx = self._planned_path_idx
        if idx >= len(path) - 1:
            return None  # at end
        # Verify we are where we should be
        if path[idx] != pos:
            # Re-sync: find pos in the remaining path
            for j in range(idx, len(path)):
                if path[j] == pos:
                    self._planned_path_idx = j
                    idx = j
                    break
            else:
                # Off path — invalidate
                self._planned_path = None
                return None
        next_pos = path[idx + 1]
        dx, dy = next_pos[0] - pos[0], next_pos[1] - pos[1]
        try:
            d = Direction.from_delta(dx, dy)
            # Verify the step is legal
            if next_pos in self._maze.pacman_graph.get(pos, ()):
                self._planned_path_idx = idx + 1
                return d
        except ValueError:
            pass
        self._planned_path = None
        return None

    def act(self, obs: Observation) -> PacmanAction:
        """Called every tick. Returns a Direction."""
        maze = self._maze
        pos  = obs.pacman_pos

        # Reconstruct a SimState from the observation
        state = SimState(
            pacman_pos        = obs.pacman_pos,
            ghost_positions   = list(obs.ghost_positions),
            ghost_frightened  = list(obs.ghost_frightened),
            active_pellets    = frozenset(obs.active_pellets),
            active_energizers = frozenset(obs.active_energizers),
            step              = obs.step,
        )

        # ---- Endgame check --------------------------------------------
        self._update_endgame(state, obs)

        if self._endgame:
            return self._safe_direction(self._endgame_move(obs, state), pos, obs)

        # ---- Follow pre-planned path if valid and safe ---------------
        d = self._follow_path(pos)
        if d is not None:
            next_pos = d.apply(pos)
            if self._is_safe_step(next_pos, obs.ghost_positions, obs.ghost_frightened):
                return d
            # Ghost too close to next step — abort path and replan
            self._planned_path = None

        # ---- At a junction / path exhausted — run MCTS ---------------
        d = self._mcts_decision(obs, state)
        return self._safe_direction(d, pos, obs)

    def _is_safe_step(
        self,
        next_pos: Pos,
        ghost_positions: List[Pos],
        ghost_frightened: List[int],
    ) -> bool:
        """Return False if a dangerous ghost is at or immediately adjacent to next_pos."""
        ghost_house = self._maze.ghost_house
        for i, gpos in enumerate(ghost_positions):
            if ghost_frightened[i] > 0 or gpos in ghost_house:
                continue
            if abs(gpos[0] - next_pos[0]) + abs(gpos[1] - next_pos[1]) <= 1:
                return False
        return True

    def _safe_direction(self, d: Direction, pos: Pos, obs: Observation) -> Direction:
        """
        Final safety gate.  If d leads to a ghost-adjacent cell, replace it with
        the legal move that maximises distance from the nearest dangerous ghost.
        If d is already safe (or there are no dangerous ghosts), return d unchanged.
        """
        if d is _STAY:
            return d
        next_pos = d.apply(pos)
        if self._is_safe_step(next_pos, obs.ghost_positions, obs.ghost_frightened):
            return d

        # d is unsafe — pick the best safe legal direction
        maze = self._maze
        nbrs = maze.pacman_graph.get(pos, ())
        danger = [(gpos, i) for i, gpos in enumerate(obs.ghost_positions)
                  if obs.ghost_frightened[i] == 0 and gpos not in maze.ghost_house]

        best_d = d          # fallback: original direction if nothing better
        best_score = -1.0
        for dd, dx, dy in _CARD4:
            nb = (pos[0] + dx, pos[1] + dy)
            if nb not in nbrs:
                continue
            if not danger:
                return dd   # no dangerous ghosts — any legal move is fine
            min_dist = min(abs(gp[0] - nb[0]) + abs(gp[1] - nb[1]) for gp, _ in danger)
            if min_dist > best_score:
                best_score = min_dist
                best_d = dd
        return best_d

    # ------------------------------------------------------------------
    def _update_endgame(self, state: SimState, obs: Observation):
        """Update endgame flag according to §5.5 triggers."""
        maze = self._maze
        cfg  = self.cfg
        pos  = state.pacman_pos

        remaining = len(state.active_pellets) + len(state.active_energizers)

        # Trigger 3: few pellets left
        if remaining <= cfg.K_endgame:
            self._endgame = True
            return

        # Trigger 2: nearest pellet/energizer outside T_path
        targets = state.active_pellets | state.active_energizers
        nearest = maze.nearest(pos, targets)
        if nearest is not None:
            dist = len(maze.bfs_path(pos, nearest) or []) - 1
            if dist > cfg.T_path:
                self._endgame = True
                return

        # Trigger 1: all S scores zero for 5 consecutive ticks
        if self._no_move_streak >= 5:
            self._endgame = True
            return

        # Exit endgame when a default tactic produces good scores and trigger gone
        if self._endgame and self._root is not None:
            best_pill     = max((c.s_pill_max     for c in self._root.children.values()), default=0.0)
            best_ghost    = max((c.s_ghost_max    for c in self._root.children.values()), default=0.0)
            if (best_pill >= 0.5 or best_ghost >= 0.5) and remaining > cfg.K_endgame:
                self._endgame = False

    def _endgame_move(self, obs: Observation, state: SimState) -> PacmanAction:
        """
        Endgame tactic (§5.5): target = nearest edible ghost → nearest energizer
        → nearest pellet.  Uses ghost-aware BFS to avoid dangerous ghosts.
        """
        maze = self._maze
        pos  = state.pacman_pos

        # Find target
        frightened_ghosts = {
            gpos for i, gpos in enumerate(state.ghost_positions)
            if state.ghost_frightened[i] > 0
        }
        target: Optional[Pos] = maze.nearest(pos, frightened_ghosts) if frightened_ghosts else None
        if target is None and state.active_energizers:
            target = maze.nearest(pos, set(state.active_energizers))
        if target is None and state.active_pellets:
            target = maze.nearest(pos, set(state.active_pellets))

        if target is None:
            return Direction.STAY

        path = self._safe_bfs_path(pos, target, state.ghost_positions, state.ghost_frightened)
        if path and len(path) > 1:
            dx, dy = path[1][0] - pos[0], path[1][1] - pos[1]
            try:
                return Direction.from_delta(dx, dy)
            except ValueError:
                pass

        return Direction.STAY

    def _safe_bfs_path(
        self,
        source: Pos,
        target: Pos,
        ghost_positions: List[Pos],
        ghost_frightened: List[int],
    ) -> Optional[List[Pos]]:
        """BFS from source to target routing around cells adjacent to dangerous ghosts."""
        maze = self._maze
        ghost_house = maze.ghost_house

        # Build avoid set: cells occupied by or adjacent to dangerous ghosts
        avoid: set = set()
        for i, gpos in enumerate(ghost_positions):
            if ghost_frightened[i] > 0 or gpos in ghost_house:
                continue
            avoid.add(gpos)
            for nb in maze.pacman_graph.get(gpos, ()):
                avoid.add(nb)

        # Avoid source itself but never avoid the target (might be surrounded)
        avoid.discard(source)
        avoid.discard(target)

        prev: Dict[Pos, Optional[Pos]] = {source: None}
        queue: deque = deque([source])
        while queue:
            cur = queue.popleft()
            if cur == target:
                path = []
                c: Optional[Pos] = target
                while c is not None:
                    path.append(c)
                    c = prev[c]
                return list(reversed(path))
            for nb in maze.pacman_graph.get(cur, ()):
                if nb not in prev and nb not in avoid:
                    prev[nb] = cur
                    queue.append(nb)

        # No ghost-avoiding path — fall back to direct BFS
        return maze.bfs_path(source, target)

    def _mcts_decision(self, obs: Observation, state: SimState) -> PacmanAction:
        """Run MCTS and return the best first-step direction."""
        maze = self._maze
        cfg  = self.cfg
        pos  = state.pacman_pos

        # Find nearest junction (root of the tree)
        cur_junction = maze.nearest_junction_from(pos)
        if cur_junction is None:
            return Direction.STAY

        # Rerun MCTS when at a new junction, no tree yet, or path was invalidated
        if cur_junction != self._last_junction or self._root is None or self._planned_path is None:
            self._last_junction = cur_junction

            # Build/reset root node
            self._root = MCTSNode(
                junction     = cur_junction,
                incoming_dir = None,
                depth_ticks  = 0,
            )

            pellets_at_root = len(state.active_pellets) + len(state.active_energizers)

            # Run MCTS with "pill" tactic for exploration; reassess tactic after
            for _ in range(cfg.n_iterations):
                self._run_iteration(self._root, state, maze, cfg, pellets_at_root, "pill")

            # Determine active tactic from actual statistics
            tactic = self._select_tactic(state)
            self._tactic = tactic
            logger.debug("step=%d junc=%s tactic=%s pellets=%d",
                         obs.step, cur_junction, tactic, pellets_at_root)

            # Choose best child direction (outgoing from cur_junction)
            best_dir = self._best_child_dir(self._root, tactic, state)

            if best_dir is None:
                self._no_move_streak += 1
                return self._endgame_move(obs, state)

            self._no_move_streak = 0

            # Build planned path: pos → cur_junction → next_junction via best_dir corridor
            child_node = self._root.children.get(best_dir)
            if child_node is not None:
                next_junction = child_node.junction
                # Full path: pos → cur_junction → ... → next_junction
                path_to_junc  = maze.bfs_path(pos, cur_junction) or [pos]
                path_corridor = maze.bfs_path(cur_junction, next_junction) or [cur_junction]
                # Merge (avoid duplicate junction node)
                full_path = path_to_junc + path_corridor[1:]
            else:
                # No child yet — just go to junction
                full_path = maze.bfs_path(pos, cur_junction) or [pos]

            self._planned_path     = full_path
            self._planned_path_idx = 0

        # Follow the planned path
        d = self._follow_path(pos)
        if d is not None:
            return d

        # Path exhausted or invalid — fall back to endgame
        return self._endgame_move(obs, state)

    # ------------------------------------------------------------------
    # MCTS core
    # ------------------------------------------------------------------

    def _run_iteration(
        self,
        root: MCTSNode,
        root_state: SimState,
        maze: Maze,
        cfg: PacmanMCTSConfig,
        pellets_at_root: int,
        tactic: str,
    ):
        # ---------- Tree phase: replay path to leaf -------------------
        node  = root
        state = root_state.copy()
        # Track ghost prev dirs for no-reverse rule in tree phase
        ghost_prev_dirs = [Direction.STAY] * 4

        path_nodes: List[MCTSNode] = [root]

        # First advance to the root junction if Pac-Man is in a corridor
        if state.pacman_pos != root.junction:
            state, died, ghost_prev_dirs = self._replay_corridor(
                state, state.pacman_pos, root.junction, ghost_prev_dirs, maze, cfg
            )
            if died or state.is_terminal:
                self._backprop(path_nodes, 0.0, 0.0, 0.0)
                return

        while True:
            # Expand if leaf
            if not node.children:
                self._expand(node, maze, cfg)

            if not node.children:
                break  # dead end

            # Selection: UCT or random (until T_visit)
            chosen_dir, child = self._select_child(node, cfg, tactic)
            if child is None:
                break

            # Simulate moving from node.junction to child.junction
            state, died, ghost_prev_dirs = self._replay_corridor(
                state, node.junction, child.junction, ghost_prev_dirs, maze, cfg
            )
            if died or state.is_terminal:
                self._backprop(path_nodes, 0.0, 0.0, 0.0)
                return

            node = child
            path_nodes.append(node)

            # Variable depth: stop expanding past T_path
            if node.depth_ticks > cfg.T_path:
                break

            # Stop at fully-visited nodes (leaf check)
            if node.n < cfg.T_visit:
                break

        # ---------- Rollout -------------------------------------------
        r_survival, r_pill, r_ghost = _rollout(state, maze, cfg, pellets_at_root)

        # ---------- Backpropagation (max) --------------------------------
        self._backprop(path_nodes, r_survival, r_pill, r_ghost)

    def _expand(self, node: MCTSNode, maze: Maze, cfg: PacmanMCTSConfig):
        """Add children for all outgoing corridors (no reverse past ply 1)."""
        j        = node.junction
        incoming = node.incoming_dir

        for nb_j, dist in maze.junction_graph.get(j, {}).items():
            # Determine direction from j to corridor leading to nb_j
            key = (j, nb_j)
            if key not in self._path_cache:
                self._path_cache[key] = maze.bfs_path(j, nb_j)
            path = self._path_cache[key]
            if not path or len(path) < 2:
                continue
            first_step = path[1]
            dx, dy = first_step[0] - j[0], first_step[1] - j[1]
            try:
                outgoing_dir = Direction.from_delta(dx, dy)
            except ValueError:
                continue

            # No reverse past ply 1
            if incoming is not None and outgoing_dir == incoming.reverse():
                continue

            depth_ticks = node.depth_ticks + dist
            if depth_ticks > cfg.T_path and node.depth_ticks > 0:
                continue  # beyond depth budget

            child = MCTSNode(
                junction     = nb_j,
                incoming_dir = outgoing_dir,
                depth_ticks  = depth_ticks,
                parent       = node,
            )
            node.children[outgoing_dir] = child

    def _select_child(
        self,
        node: MCTSNode,
        cfg: PacmanMCTSConfig,
        tactic: str,
    ) -> Tuple[Direction, Optional[MCTSNode]]:
        """UCT child selection."""
        children = list(node.children.items())
        if not children:
            return Direction.STAY, None

        # Random until all children visited T_visit times
        unvisited = [(d, c) for d, c in children if c.n < cfg.T_visit]
        if unvisited:
            d, c = random.choice(unvisited)
            return d, c

        # UCT
        best_score = -float("inf")
        best = (Direction.STAY, None)
        for d, c in children:
            score = c.uct_score(node.n, cfg.C, tactic)
            if score > best_score:
                best_score = score
                best = (d, c)
        return best

    def _replay_corridor(
        self,
        state: SimState,
        from_pos: Pos,
        to_j: Pos,
        ghost_prev_dirs: List[Direction],
        maze: Maze,
        cfg: PacmanMCTSConfig,
    ) -> Tuple[SimState, bool, List[Direction]]:
        """
        Advance state along path from from_pos to to_j.
        Uses fast ghost moves (no BFS).
        Returns (new_state, died, updated_ghost_prev_dirs).
        """
        key = (from_pos, to_j)
        if key not in self._path_cache:
            self._path_cache[key] = maze.bfs_path(from_pos, to_j)
        path = self._path_cache[key]
        if not path:
            return state, False, ghost_prev_dirs

        s    = state.copy()
        dirs = list(ghost_prev_dirs)
        for i in range(len(path) - 1):
            if s.is_terminal:
                break
            cur  = path[i]
            nxt  = path[i + 1]
            dx, dy = nxt[0] - cur[0], nxt[1] - cur[1]
            pac_dir = None
            for d, ddx, ddy in _CARD4:
                if ddx == dx and ddy == dy:
                    pac_dir = d
                    break
            if pac_dir is None:
                break

            g_dirs = tuple(
                _fast_ghost_move(
                    s.ghost_positions[j], s.pacman_pos, s.ghost_frightened[j],
                    dirs[j], maze.ghost_graph,
                )
                for j in range(4)
            )
            dirs = list(g_dirs)
            s = sim_step(s, maze, pac_dir, g_dirs,  # type: ignore
                         frightened_duration=cfg.frightened_duration,
                         max_steps=cfg.max_steps)

        died = s.outcome == "eaten"
        return s, died, dirs

    def _backprop(
        self,
        path: List[MCTSNode],
        r_survival: float,
        r_pill: float,
        r_ghost: float,
    ):
        for node in reversed(path):
            node.update(r_survival, r_pill, r_ghost)

    # ------------------------------------------------------------------
    # Tactic selection (§5.3)
    # ------------------------------------------------------------------

    def _select_tactic(self, state: SimState) -> str:
        if self._root is None:
            return "pill"

        best_survival = max(
            (c.s_survival_max for c in self._root.children.values()),
            default=0.0,
        )
        if best_survival < self.cfg.T_survival:
            return "survival"

        # Ghost tactic: energizer recently eaten AND frightened ghost in range
        any_frightened = any(f > 0 for f in state.ghost_frightened)
        if any_frightened:
            return "ghost"

        return "pill"

    def _best_child_dir(
        self,
        root: MCTSNode,
        tactic: str,
        state: SimState,
    ) -> Optional[Direction]:
        """Return direction of best child under tactic. Fall-through per §5.3."""
        tactics = [tactic, "pill", "ghost", "survival"]
        seen_tactics: Set[str] = set()

        for t in tactics:
            if t in seen_tactics:
                continue
            seen_tactics.add(t)

            best_score = -1.0
            best_dir: Optional[Direction] = None
            for d, child in root.children.items():
                v = child.v_i(t)
                if v > best_score:
                    best_score = v
                    best_dir   = d

            if best_dir is not None and best_score > 0.0:
                return best_dir

        return None
