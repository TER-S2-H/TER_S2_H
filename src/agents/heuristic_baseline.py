"""
Heuristic agents used both as standalone baselines and as MCTS rollout policies.

HeuristicPacman (§5.4 rollout policy):
  - LGR (Last-Good-Reply): repeat last move that didn't lead to death
  - Prefer uncleared safe corridors
  - Avoid non-frightened ghosts
  - Chase frightened ghosts

HeuristicGhosts (§6.4 / paper §IV-F):
  - Case 1: chase Pac-Man (when not frightened)
  - Case 2: flee from Pac-Man (when frightened)
  - Case 3: no two ghosts on the same edge same direction
  - ε-greedy with epsilon_ghost = 0.05
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple

from ..game.maze import Maze
from ..game.types import Direction, GhostAction, Observation, PacmanAction, Pos


# ---------------------------------------------------------------------------
# Shared BFS utility (independent of Maze to be usable inside rollouts)
# ---------------------------------------------------------------------------

def _bfs_next_step(
    start: Pos,
    target: Pos,
    graph: Dict[Pos, FrozenSet[Pos]],
    forbidden_dir: Optional[Direction] = None,
) -> Optional[Direction]:
    """Return the first-step direction on the BFS shortest path from start to target."""
    if start == target:
        return Direction.STAY
    prev: Dict[Pos, Optional[Pos]] = {start: None}
    queue = deque([start])
    while queue:
        pos = queue.popleft()
        for nb in graph.get(pos, ()):
            if nb not in prev:
                prev[nb] = pos
                if nb == target:
                    # Trace back to find first step
                    cur = nb
                    while prev[cur] != start:
                        cur = prev[cur]  # type: ignore
                    dx, dy = cur[0] - start[0], cur[1] - start[1]
                    try:
                        d = Direction.from_delta(dx, dy)
                        if d == forbidden_dir:
                            continue
                        return d
                    except ValueError:
                        pass
                queue.append(nb)
    return None


def _ghost_danger(
    pos: Pos,
    ghost_positions: List[Pos],
    ghost_frightened: List[int],
    graph: Dict[Pos, FrozenSet[Pos]],
    danger_radius: int = 5,
) -> bool:
    """True if a non-frightened ghost is within danger_radius BFS steps of pos."""
    visited = {pos}
    queue = deque([(pos, 0)])
    while queue:
        p, d = queue.popleft()
        if d >= danger_radius:
            continue
        for nb in graph.get(p, ()):
            if nb not in visited:
                visited.add(nb)
                for i, gpos in enumerate(ghost_positions):
                    if gpos == nb and ghost_frightened[i] == 0:
                        return True
                queue.append((nb, d + 1))
    return False


# ---------------------------------------------------------------------------
# Heuristic Pac-Man
# ---------------------------------------------------------------------------

class HeuristicPacman:
    """
    Heuristic Pac-Man agent — also the MCTS rollout policy.
    Uses LGR (Last Good Reply) as its primary mechanism.
    """
    name = "heuristic_pacman"

    def reset(self, maze: Maze, role: Literal["pacman", "ghosts"]) -> None:
        self._maze = maze
        self._lgr: Optional[Direction] = None   # Last-Good-Reply

    def act(self, obs: Observation) -> PacmanAction:
        return self._choose(
            pos               = obs.pacman_pos,
            ghost_positions   = obs.ghost_positions,
            ghost_frightened  = obs.ghost_frightened,
            active_pellets    = set(obs.active_pellets),
            active_energizers = set(obs.active_energizers),
        )

    def _choose(
        self,
        pos: Pos,
        ghost_positions: List[Pos],
        ghost_frightened: List[int],
        active_pellets: Set[Pos],
        active_energizers: Set[Pos],
    ) -> Direction:
        maze    = self._maze
        graph   = maze.pacman_graph
        targets = set(active_pellets) | set(active_energizers)

        # Frightened ghosts are high-value targets
        frightened_targets = {
            gpos for i, gpos in enumerate(ghost_positions)
            if ghost_frightened[i] > 0
        }

        # Choose target: frightened ghost > energizer > nearest pellet
        target: Optional[Pos] = None
        if frightened_targets:
            target = maze.nearest(pos, frightened_targets)
        if target is None:
            if active_energizers and not any(f > 0 for f in ghost_frightened):
                # Consider energizer only if no ghosts currently frightened
                target = maze.nearest(pos, set(active_energizers))
        if target is None:
            target = maze.nearest(pos, targets)

        if target is None:
            return Direction.STAY

        # Build a safe direction set (avoid non-frightened ghosts)
        danger_zone: Set[Pos] = {
            gpos for i, gpos in enumerate(ghost_positions)
            if ghost_frightened[i] == 0
        }

        # Try LGR first (repeat last move that was safe)
        if self._lgr is not None:
            nb = self._lgr.apply(pos)
            if nb in graph.get(pos, ()):
                is_safe = nb not in danger_zone and not _ghost_danger(
                    nb, ghost_positions, ghost_frightened, graph, danger_radius=3
                )
                if is_safe:
                    return self._lgr

        # BFS path to target
        path = maze.bfs_path(pos, target)
        if path and len(path) > 1:
            next_pos = path[1]
            dx, dy   = next_pos[0] - pos[0], next_pos[1] - pos[1]
            try:
                d = Direction.from_delta(dx, dy)
                nb = d.apply(pos)
                if nb in graph.get(pos, ()):
                    is_safe = nb not in danger_zone and not _ghost_danger(
                        nb, ghost_positions, ghost_frightened, graph, danger_radius=3
                    )
                    if is_safe:
                        self._lgr = d
                        return d
            except ValueError:
                pass

        # Fallback: any safe legal move
        legal_safe = [
            d for d in Direction.cardinals()
            if d.apply(pos) in graph.get(pos, ())
            and d.apply(pos) not in danger_zone
            and not _ghost_danger(d.apply(pos), ghost_positions, ghost_frightened, graph, 3)
        ]
        if legal_safe:
            d = random.choice(legal_safe)
            self._lgr = d
            return d

        # Last resort: any legal move
        legal = [d for d in Direction.cardinals() if d.apply(pos) in graph.get(pos, ())]
        if legal:
            return random.choice(legal)
        return Direction.STAY


# ---------------------------------------------------------------------------
# Heuristic Ghosts
# ---------------------------------------------------------------------------

class HeuristicGhosts:
    """
    Heuristic ghost-team agent — also the MCTS rollout policy for ghosts.
    Follows paper §IV-F Case 1/2/3.
    ε-greedy: with prob epsilon_ghost, make a random legal move.
    """
    name = "heuristic_ghosts"

    def __init__(self, epsilon_ghost: float = 0.05):
        self.epsilon_ghost = epsilon_ghost

    def reset(self, maze: Maze, role: Literal["pacman", "ghosts"]) -> None:
        self._maze = maze
        self._prev_dirs: List[Direction] = [Direction.STAY] * 4

    def act(self, obs: Observation) -> GhostAction:
        return self._choose(
            pacman_pos       = obs.pacman_pos,
            ghost_positions  = obs.ghost_positions,
            ghost_frightened = obs.ghost_frightened,
        )

    def _choose(
        self,
        pacman_pos: Pos,
        ghost_positions: List[Pos],
        ghost_frightened: List[int],
    ) -> GhostAction:
        maze    = self._maze
        graph   = maze.ghost_graph
        chosen  = [Direction.STAY] * 4
        edges_taken: Set[Tuple[Pos, Pos]] = set()  # Case 3: no duplicate ghost edges

        for i, gpos in enumerate(ghost_positions):
            reverse = self._prev_dirs[i].reverse()

            # ε-greedy
            if random.random() < self.epsilon_ghost:
                legal = [
                    d for d in Direction.cardinals()
                    if d.apply(gpos) in graph.get(gpos, ())
                    and d != reverse
                ]
                if not legal:
                    legal = [d for d in Direction.cardinals() if d.apply(gpos) in graph.get(gpos, ())]
                d = random.choice(legal) if legal else Direction.STAY
            else:
                frightened = ghost_frightened[i] > 0

                if frightened:
                    # Case 2: flee — move away from Pac-Man
                    d = self._flee(gpos, pacman_pos, reverse, graph)
                else:
                    # Case 1: chase — BFS toward Pac-Man
                    d = self._chase(gpos, pacman_pos, reverse, graph)

                # Case 3: avoid duplicating an occupied edge
                nb = d.apply(gpos)
                edge = (min(gpos, nb), max(gpos, nb))
                if edge in edges_taken:
                    # Try another direction
                    alts = [
                        dd for dd in Direction.cardinals()
                        if dd.apply(gpos) in graph.get(gpos, ())
                        and dd != reverse
                        and (min(gpos, dd.apply(gpos)), max(gpos, dd.apply(gpos))) not in edges_taken
                    ]
                    if alts:
                        d = alts[0]
                        nb = d.apply(gpos)
                        edge = (min(gpos, nb), max(gpos, nb))

            chosen[i] = d
            nb = d.apply(gpos)
            edge = (min(gpos, nb), max(gpos, nb))
            edges_taken.add(edge)
            self._prev_dirs[i] = d

        return tuple(chosen)  # type: ignore

    def _chase(
        self,
        gpos: Pos,
        pacman_pos: Pos,
        reverse: Direction,
        graph: Dict[Pos, FrozenSet[Pos]],
    ) -> Direction:
        """BFS chase: move toward Pac-Man, no reverse."""
        d = _bfs_next_step(gpos, pacman_pos, graph, forbidden_dir=None)
        if d is not None and d != reverse:
            return d
        # Try any legal non-reverse
        legal = [dd for dd in Direction.cardinals() if dd.apply(gpos) in graph.get(gpos, ()) and dd != reverse]
        if legal:
            return min(legal, key=lambda dd: (dd.apply(gpos)[0] - pacman_pos[0])**2 + (dd.apply(gpos)[1] - pacman_pos[1])**2)
        # Dead end: reverse allowed
        legal = [dd for dd in Direction.cardinals() if dd.apply(gpos) in graph.get(gpos, ())]
        return legal[0] if legal else Direction.STAY

    def _flee(
        self,
        gpos: Pos,
        pacman_pos: Pos,
        reverse: Direction,
        graph: Dict[Pos, FrozenSet[Pos]],
    ) -> Direction:
        """Flee: move away from Pac-Man."""
        legal = [dd for dd in Direction.cardinals() if dd.apply(gpos) in graph.get(gpos, ()) and dd != reverse]
        if not legal:
            legal = [dd for dd in Direction.cardinals() if dd.apply(gpos) in graph.get(gpos, ())]
        if not legal:
            return Direction.STAY
        return max(legal, key=lambda dd: (dd.apply(gpos)[0] - pacman_pos[0])**2 + (dd.apply(gpos)[1] - pacman_pos[1])**2)
