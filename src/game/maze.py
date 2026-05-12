"""
Maze loading, graph extraction, and all-pairs shortest-path on the junction graph.

Key design principle from §3.4 and §8:
  pacman_graph  — excludes ghost_wall cells and ghost-house interior
  ghost_graph   — includes everything (ghost_walls + ghost-house interior)

Ghost-house interior is inferred by flood-fill from ghost_wall cells into cells
that are neither in the regular node set nor classified as walls.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .types import Direction, Pos

logger = logging.getLogger(__name__)

# Default parameters (can be overridden in config)
DEFAULT_PACMAN_SPAWN: Pos = (13, 23)
FRIGHTENED_DURATION: int  = 40   # ticks


class Maze:
    """
    Loaded from a maze*.json file.  Builds all graphs and caches BFS distances.

    Attributes
    ----------
    width, height : int
    pacman_graph  : dict[Pos, frozenset[Pos]]
    ghost_graph   : dict[Pos, frozenset[Pos]]
    ghost_house   : frozenset[Pos]  — cells only ghosts can enter
    ghost_spawns  : list[Pos]       — where eaten ghosts respawn (4 positions)
    pacman_spawn  : Pos
    initial_pellets    : frozenset[Pos]
    initial_energizers : frozenset[Pos]
    junction_graph     : dict[Pos, dict[Pos, int]]  — junction→{junction: dist}
    junctions          : frozenset[Pos]  — nodes with ≥3 neighbours or dead-ends
    """

    # ------------------------------------------------------------------
    def __init__(self, data: dict, pacman_spawn: Optional[Pos] = None):
        meta   = data["metadata"]
        self.width:  int = meta["width"]
        self.height: int = meta["height"]

        # ---- Regular walkable nodes (PATH / PATHBLANK) ----------------
        pacman_nodes: Set[Pos] = set()
        pellets:      Set[Pos] = set()
        for n in data["nodes"]:
            pos = (n["x"], n["y"])
            pacman_nodes.add(pos)
            if n["type"] == "PATH":
                pellets.add(pos)

        # ---- Explicit energizer positions -----------------------------
        energizers: Set[Pos] = {(e["x"], e["y"]) for e in data.get("energizers", [])}
        # Energizer cells are PATH nodes — remove from pellets set
        pellets -= energizers

        self.initial_pellets:    FrozenSet[Pos] = frozenset(pellets)
        self.initial_energizers: FrozenSet[Pos] = frozenset(energizers)

        # ---- Wall set (nothing can enter) -----------------------------
        walls: Set[Pos] = {(w["x"], w["y"]) for w in data.get("walls", [])}

        # ---- Ghost-wall cells (door) —walkable only by ghosts ---------
        ghost_wall_cells: Set[Pos] = {(g["x"], g["y"]) for g in data.get("ghost_walls", [])}

        # ---- Ghost-house interior (flood-fill from door) -------------
        ghost_house_interior = self._find_ghost_house(
            ghost_wall_cells, pacman_nodes, walls
        )
        self.ghost_house: FrozenSet[Pos] = frozenset(ghost_house_interior | ghost_wall_cells)

        # ---- pacman_graph: PATH/PATHBLANK nodes, no ghost-house cells -
        pacman_adj: Dict[Pos, Set[Pos]] = {p: set() for p in pacman_nodes}
        for edge in data["edges"]:
            a: Pos = tuple(int(v) for v in edge["from"].split(","))  # type: ignore
            b: Pos = tuple(int(v) for v in edge["to"].split(","))    # type: ignore
            if a in pacman_adj and b in pacman_adj:
                pacman_adj[a].add(b)
                pacman_adj[b].add(a)

        self.pacman_graph: Dict[Pos, FrozenSet[Pos]] = {
            p: frozenset(nbrs) for p, nbrs in pacman_adj.items()
        }

        # ---- ghost_graph: pacman_graph + door + interior --------------
        ghost_adj: Dict[Pos, Set[Pos]] = {p: set(nbrs) for p, nbrs in pacman_adj.items()}
        ghost_all = pacman_nodes | ghost_wall_cells | ghost_house_interior
        for gp in ghost_wall_cells | ghost_house_interior:
            if gp not in ghost_adj:
                ghost_adj[gp] = set()
        for p in ghost_wall_cells | ghost_house_interior:
            x, y = p
            for d in Direction.cardinals():
                nb = (x + d.dx, y + d.dy)
                if nb in ghost_all:
                    ghost_adj[p].add(nb)
                    ghost_adj[nb].add(p)

        self.ghost_graph: Dict[Pos, FrozenSet[Pos]] = {
            p: frozenset(nbrs) for p, nbrs in ghost_adj.items()
        }

        # ---- Pac-Man spawn --------------------------------------------
        self.pacman_spawn: Pos = pacman_spawn or DEFAULT_PACMAN_SPAWN
        if self.pacman_spawn not in self.pacman_graph:
            # Fall back to centroid of pacman nodes
            logger.warning("pacman_spawn %s not in graph; using centroid", self.pacman_spawn)
            cx = sum(p[0] for p in pacman_nodes) // len(pacman_nodes)
            cy = sum(p[1] for p in pacman_nodes) // len(pacman_nodes)
            self.pacman_spawn = min(pacman_nodes, key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)

        # ---- Ghost spawn positions ------------------------------------
        self.ghost_spawns: List[Pos] = self._determine_ghost_spawns(ghost_house_interior, ghost_wall_cells)

        # ---- Junction / dead-end classification ----------------------
        junctions: Set[Pos] = set()
        for pos, nbrs in self.pacman_graph.items():
            if len(nbrs) != 2:   # dead-end (1) or junction (≥3)
                junctions.add(pos)
        self.junctions: FrozenSet[Pos] = frozenset(junctions)

        # ---- Junction graph (weighted corridors) ---------------------
        self.junction_graph: Dict[Pos, Dict[Pos, int]] = self._build_junction_graph()

        # ---- Graph diameter (for T_path warning) --------------------
        self._graph_diameter: Optional[int] = None   # lazy

        # ---- Reachability check ------------------------------------
        self._verify_reachability()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_ghost_house(
        self,
        door_cells: Set[Pos],
        pacman_nodes: Set[Pos],
        walls: Set[Pos],
    ) -> Set[Pos]:
        """Flood-fill from door cells into cells that pacman cannot reach."""
        interior: Set[Pos] = set()
        queue = deque(door_cells)
        visited: Set[Pos] = set(door_cells)

        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (x + dx, y + dy)
                if nb in visited:
                    continue
                if nb in walls or nb in pacman_nodes:
                    continue
                if not (0 <= nb[0] < self.width and 0 <= nb[1] < self.height):
                    continue
                interior.add(nb)
                visited.add(nb)
                queue.append(nb)

        return interior

    def _determine_ghost_spawns(
        self,
        interior: Set[Pos],
        door_cells: Set[Pos],
    ) -> List[Pos]:
        """Choose 4 representative spawn positions inside the ghost house."""
        if not interior:
            logger.warning("Ghost house interior is empty; using pacman_spawn as fallback")
            return [self.pacman_spawn] * 4

        # Sort by (y, x) so we get a reproducible ordering
        cells = sorted(interior)
        n = len(cells)
        # Pick 4 evenly spaced cells
        indices = [n // 5, n * 2 // 5, n * 3 // 5, n * 4 // 5]
        spawns = [cells[i] for i in indices]
        # Ensure 4 distinct positions
        while len(spawns) < 4:
            spawns.append(spawns[-1])
        return spawns[:4]

    def _build_junction_graph(self) -> Dict[Pos, Dict[Pos, int]]:
        """
        Build junction graph: nodes are junctions/dead-ends,
        edges are corridor paths weighted by length.
        """
        jg: Dict[Pos, Dict[Pos, int]] = {j: {} for j in self.junctions}

        for start in self.junctions:
            for nb in self.pacman_graph[start]:
                # Walk corridor from start through nb until next junction
                prev, cur = start, nb
                dist = 1
                while cur not in self.junctions:
                    nexts = [n for n in self.pacman_graph[cur] if n != prev]
                    if not nexts:
                        break
                    prev, cur = cur, nexts[0]
                    dist += 1
                if cur in self.junctions and cur != start:
                    # Only set if not already set (handles multi-path between same junctions)
                    if cur not in jg[start] or jg[start][cur] > dist:
                        jg[start][cur] = dist
                    if start not in jg[cur] or jg[cur][start] > dist:
                        jg[cur][start] = dist

        return jg

    def _verify_reachability(self):
        """Every PATH cell must be reachable from pacman_spawn via pacman_graph."""
        reachable = self.bfs_reachable(self.pacman_spawn, use_ghost_graph=False)
        unreachable = (self.initial_pellets | self.initial_energizers) - reachable
        if unreachable:
            raise ValueError(
                f"Maze reachability error: {len(unreachable)} pellet/energizer cells "
                f"unreachable from pacman_spawn {self.pacman_spawn} via pacman_graph. "
                "Check ghost-house exclusion (§3.4)."
            )

    # ------------------------------------------------------------------
    # BFS utilities
    # ------------------------------------------------------------------

    def bfs_distances(
        self,
        source: Pos,
        use_ghost_graph: bool = False,
    ) -> Dict[Pos, int]:
        """BFS shortest distances from source on pacman_graph (or ghost_graph)."""
        graph = self.ghost_graph if use_ghost_graph else self.pacman_graph
        dist: Dict[Pos, int] = {source: 0}
        queue = deque([source])
        while queue:
            pos = queue.popleft()
            d = dist[pos]
            for nb in graph.get(pos, ()):
                if nb not in dist:
                    dist[nb] = d + 1
                    queue.append(nb)
        return dist

    def bfs_reachable(
        self,
        source: Pos,
        use_ghost_graph: bool = False,
    ) -> FrozenSet[Pos]:
        return frozenset(self.bfs_distances(source, use_ghost_graph).keys())

    def bfs_path(
        self,
        source: Pos,
        target: Pos,
        use_ghost_graph: bool = False,
    ) -> Optional[List[Pos]]:
        """Return shortest path (list of positions) from source to target, or None."""
        graph = self.ghost_graph if use_ghost_graph else self.pacman_graph
        if source == target:
            return [source]
        prev: Dict[Pos, Optional[Pos]] = {source: None}
        queue = deque([source])
        while queue:
            pos = queue.popleft()
            for nb in graph.get(pos, ()):
                if nb not in prev:
                    prev[nb] = pos
                    if nb == target:
                        path = []
                        cur: Optional[Pos] = target
                        while cur is not None:
                            path.append(cur)
                            cur = prev[cur]
                        return list(reversed(path))
                    queue.append(nb)
        return None

    def nearest(
        self,
        source: Pos,
        candidates: Set[Pos],
        use_ghost_graph: bool = False,
    ) -> Optional[Pos]:
        """Return the candidate reachable from source with minimum BFS distance."""
        if not candidates:
            return None
        graph = self.ghost_graph if use_ghost_graph else self.pacman_graph
        queue = deque([source])
        visited: Set[Pos] = {source}
        while queue:
            pos = queue.popleft()
            if pos in candidates:
                return pos
            for nb in graph.get(pos, ()):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return None

    # ------------------------------------------------------------------
    # Junction graph APSP (Floyd-Warshall, cheap for ~30-80 nodes)
    # ------------------------------------------------------------------

    def apsp_junction(self) -> Dict[Pos, Dict[Pos, int]]:
        """All-pairs shortest path on junction_graph. Cached after first call."""
        if hasattr(self, "_apsp"):
            return self._apsp   # type: ignore

        jnodes = list(self.junctions)
        INF = 10**9
        dist: Dict[Pos, Dict[Pos, int]] = {
            j: {j2: (0 if j2 == j else self.junction_graph.get(j, {}).get(j2, INF))
                for j2 in jnodes}
            for j in jnodes
        }
        for k in jnodes:
            for i in jnodes:
                if dist[i][k] == INF:
                    continue
                for j in jnodes:
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        self._apsp: Dict[Pos, Dict[Pos, int]] = dist
        return dist

    # ------------------------------------------------------------------
    # Helpers used by MCTS
    # ------------------------------------------------------------------

    def nearest_junction_from(self, pos: Pos) -> Optional[Pos]:
        """BFS on pacman_graph to find nearest junction/dead-end."""
        if pos in self.junctions:
            return pos
        visited = {pos}
        queue = deque([pos])
        while queue:
            p = queue.popleft()
            for nb in self.pacman_graph.get(p, ()):
                if nb in self.junctions:
                    return nb
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return None

    def junction_dist(self, a: Pos, b: Pos) -> int:
        """APSP distance between two junctions. INF if unreachable."""
        apsp = self.apsp_junction()
        return apsp.get(a, {}).get(b, 10**9)

    def graph_diameter(self) -> int:
        """Maximum BFS distance between any two reachable nodes on pacman_graph."""
        if self._graph_diameter is not None:
            return self._graph_diameter
        max_d = 0
        for pos in self.pacman_graph:
            dists = self.bfs_distances(pos)
            if dists:
                max_d = max(max_d, max(dists.values()))
        self._graph_diameter = max_d
        return max_d

    def direction_to_neighbor(self, frm: Pos, to: Pos) -> Optional[Direction]:
        """Cardinal direction from frm to adjacent cell to."""
        dx, dy = to[0] - frm[0], to[1] - frm[1]
        try:
            return Direction.from_delta(dx, dy)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def load(path: str, pacman_spawn: Optional[Pos] = None) -> "Maze":
        with open(path) as f:
            data = json.load(f)
        return Maze(data, pacman_spawn=pacman_spawn)
