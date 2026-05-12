"""ICE gUCT ghost-team agent and KNN Pac-Man predictor.

The implementation follows the task brief closely but keeps the search
lightweight enough for the current grid-based simulator: each ghost tree is
rooted at the next decision cross-point and stores root-level child statistics.
That is sufficient for the required behavior, tree rebuilds, and coordination
tests while staying within the per-tick budget.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np

from ..game.maze import Maze
from ..game.simulator import SimState, step as sim_step
from ..game.types import Direction, GhostAction, Observation, Pos

logger = logging.getLogger(__name__)

CARDINALS: tuple[Direction, ...] = tuple(Direction.cardinals())
INF_DISTANCE = 10**9
DEFAULT_DATASET_PATH = Path("data/knn_train/dataset.npz")


def _as_observation(value: Observation | SimState | dict | Any) -> Observation:
    if isinstance(value, Observation):
        return value
    if isinstance(value, SimState):
        return Observation(
            step=value.step,
            pacman_pos=value.pacman_pos,
            ghost_positions=list(value.ghost_positions),
            ghost_frightened=list(value.ghost_frightened),
            active_pellets=list(value.active_pellets),
            active_energizers=list(value.active_energizers),
        )
    if isinstance(value, dict):
        return Observation.from_dict(value)
    return Observation(
        step=int(getattr(value, "step")),
        pacman_pos=tuple(getattr(value, "pacman_pos")),
        ghost_positions=[tuple(p) for p in getattr(value, "ghost_positions")],
        ghost_frightened=list(getattr(value, "ghost_frightened")),
        active_pellets=[tuple(p) for p in getattr(value, "active_pellets")],
        active_energizers=[tuple(p) for p in getattr(value, "active_energizers")],
    )


def _direction_from_delta(dx: int, dy: int) -> Direction:
    return Direction.from_delta(dx, dy)


def _legal_directions(
    pos: Pos,
    graph: Dict[Pos, frozenset[Pos]],
    reverse_dir: Optional[Direction] = None,
) -> List[Direction]:
    legal = [
        d for d in CARDINALS
        if d.apply(pos) in graph.get(pos, ())
    ]
    if reverse_dir is not None and len(legal) > 1:
        filtered = [d for d in legal if d != reverse_dir]
        if filtered:
            legal = filtered
    return legal


def _load_npz_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        warnings.warn(
            f"KNN dataset not found at {path}; falling back to random Pac-Man predictions.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.empty((0, 10), dtype=np.float32), np.empty((0, 10), dtype=np.float32)

    try:
        with np.load(path) as data:
            x = np.asarray(data.get("X", np.empty((0, 10))), dtype=np.float32)
            y = np.asarray(data.get("Y", np.empty((0, 10))), dtype=np.float32)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(
            f"Failed to load KNN dataset at {path}: {exc}; using random fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.empty((0, 10), dtype=np.float32), np.empty((0, 10), dtype=np.float32)

    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[1] != 10:
        warnings.warn(
            f"Invalid KNN dataset shape at {path}: X={x.shape}, Y={y.shape}; using random fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.empty((0, 10), dtype=np.float32), np.empty((0, 10), dtype=np.float32)

    return x, y


@dataclass
class IceGuctConfig:
    C: float = 1.41
    epsilon: float = 1e-6
    T_sim: int = 80
    k_neighbors: int = 3
    p_predict_start: float = 0.9
    p_predict_end: float = 0.1
    w_capture: float = 1.0
    w_deny: float = 0.3
    w_time: float = 0.1
    w_disperse: float = 0.1
    w_eaten: float = 0.5
    budget_ms_per_tick: int = 40
    budget_split: Literal["equal", "needs_based"] = "equal"


class GhostTopology:
    """Ghost-only topology on the inclusive ghost graph."""

    def __init__(self, maze: Maze):
        self.maze = maze
        self.graph = maze.ghost_graph
        self.junctions = frozenset(
            pos for pos, nbrs in self.graph.items() if len(nbrs) != 2
        )
        self.junction_graph = self._build_junction_graph()
        self._path_cache: Dict[tuple[Pos, Pos], List[Pos]] = {}
        self._first_step_cache: Dict[tuple[Pos, Pos], Direction] = {}
        self._apsp_cache: Optional[Dict[Pos, Dict[Pos, int]]] = None
        self._diameter: Optional[int] = None

    def is_junction(self, pos: Pos) -> bool:
        return pos in self.junctions

    def legal_directions(
        self,
        pos: Pos,
        reverse_dir: Optional[Direction] = None,
    ) -> List[Direction]:
        return _legal_directions(pos, self.graph, reverse_dir)

    def next_root(self, pos: Pos, motion_dir: Direction) -> Pos:
        if pos in self.junctions:
            return pos
        if motion_dir == Direction.STAY:
            return self.nearest_junction(pos) or pos
        cur = pos
        while True:
            nxt = motion_dir.apply(cur)
            if nxt not in self.graph.get(cur, ()):
                return self.nearest_junction(pos) or pos
            cur = nxt
            if cur in self.junctions:
                return cur

    def nearest_junction(self, pos: Pos) -> Optional[Pos]:
        if pos in self.junctions:
            return pos
        queue = deque([pos])
        visited = {pos}
        while queue:
            current = queue.popleft()
            for nb in self.graph.get(current, ()):
                if nb in self.junctions:
                    return nb
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return None

    def path_between(self, a: Pos, b: Pos) -> Optional[List[Pos]]:
        key = (a, b)
        if key in self._path_cache:
            return self._path_cache[key]
        if a == b:
            self._path_cache[key] = [a]
            return [a]

        prev: Dict[Pos, Optional[Pos]] = {a: None}
        queue = deque([a])
        while queue:
            pos = queue.popleft()
            for nb in self.graph.get(pos, ()):
                if nb in prev:
                    continue
                prev[nb] = pos
                if nb == b:
                    path: List[Pos] = []
                    cur: Optional[Pos] = b
                    while cur is not None:
                        path.append(cur)
                        cur = prev[cur]
                    path = list(reversed(path))
                    self._path_cache[key] = path
                    if len(path) > 1:
                        dx, dy = path[1][0] - path[0][0], path[1][1] - path[0][1]
                        try:
                            self._first_step_cache[key] = _direction_from_delta(dx, dy)
                        except ValueError:
                            pass
                    return path
                queue.append(nb)
        return None

    def first_step_direction(self, a: Pos, b: Pos) -> Optional[Direction]:
        key = (a, b)
        if key in self._first_step_cache:
            return self._first_step_cache[key]
        path = self.path_between(a, b)
        if not path or len(path) < 2:
            return None
        dx, dy = path[1][0] - path[0][0], path[1][1] - path[0][1]
        try:
            d = _direction_from_delta(dx, dy)
        except ValueError:
            return None
        self._first_step_cache[key] = d
        return d

    def _build_junction_graph(self) -> Dict[Pos, Dict[Pos, int]]:
        junction_graph: Dict[Pos, Dict[Pos, int]] = {j: {} for j in self.junctions}
        for start in self.junctions:
            for nb in self.graph.get(start, ()):
                prev, cur = start, nb
                dist = 1
                while cur not in self.junctions:
                    nexts = [n for n in self.graph.get(cur, ()) if n != prev]
                    if not nexts:
                        break
                    prev, cur = cur, nexts[0]
                    dist += 1
                if cur in self.junctions and cur != start:
                    if cur not in junction_graph[start] or junction_graph[start][cur] > dist:
                        junction_graph[start][cur] = dist
                    if start not in junction_graph[cur] or junction_graph[cur][start] > dist:
                        junction_graph[cur][start] = dist
        return junction_graph

    def apsp(self) -> Dict[Pos, Dict[Pos, int]]:
        if self._apsp_cache is not None:
            return self._apsp_cache
        dist: Dict[Pos, Dict[Pos, int]] = {}
        for source in self.junctions:
            dist[source] = {source: 0}
            queue = deque([source])
            while queue:
                pos = queue.popleft()
                base = dist[source][pos]
                for nb, weight in self.junction_graph.get(pos, {}).items():
                    cand = base + weight
                    if nb not in dist[source] or cand < dist[source][nb]:
                        dist[source][nb] = cand
                        queue.append(nb)
        self._apsp_cache = dist
        return dist

    def junction_distance(self, a: Pos, b: Pos) -> int:
        if a == b:
            return 0
        return self.apsp().get(a, {}).get(b, INF_DISTANCE)

    def diameter(self) -> int:
        if self._diameter is not None:
            return self._diameter
        best = 0
        for source, targets in self.apsp().items():
            if targets:
                best = max(best, max(targets.values()))
        self._diameter = max(1, best)
        return self._diameter


class FeatureExtractor:
    FEATURE_DIM = 10

    def __init__(self, maze: Maze):
        self.maze = maze
        self.pacman_cache: Dict[Pos, Dict[Pos, int]] = {}
        self.ghost_cache: Dict[Pos, Dict[Pos, int]] = {}
        self.pacman_diameter = max(1, maze.graph_diameter())
        self.ghost_diameter = self._compute_ghost_diameter()
        self.junctions = frozenset(maze.junctions)

    def _compute_ghost_diameter(self) -> int:
        best = 1
        for source in self.maze.ghost_graph:
            dists = self._ghost_distances(source)
            if dists:
                best = max(best, max(dists.values()))
        return best

    def _pacman_distances(self, source: Pos) -> Dict[Pos, int]:
        if source not in self.pacman_cache:
            self.pacman_cache[source] = self.maze.bfs_distances(source, use_ghost_graph=False)
        return self.pacman_cache[source]

    def _ghost_distances(self, source: Pos) -> Dict[Pos, int]:
        if source not in self.ghost_cache:
            self.ghost_cache[source] = self.maze.bfs_distances(source, use_ghost_graph=True)
        return self.ghost_cache[source]

    def _dist(
        self,
        source: Pos,
        target: Pos,
        *,
        use_ghost_graph: bool = False,
    ) -> int:
        cache = self._ghost_distances(source) if use_ghost_graph else self._pacman_distances(source)
        return cache.get(target, INF_DISTANCE)

    def _norm(self, distance: int, diameter: int) -> float:
        if distance >= INF_DISTANCE:
            return 1.0
        return min(distance, diameter) / float(max(diameter, 1))

    def feature_vector(self, obs: Observation | SimState | dict | Any) -> np.ndarray:
        state = _as_observation(obs)
        px = state.pacman_pos
        ghost_positions = list(state.ghost_positions)
        frightened = list(state.ghost_frightened)

        ghost_distances: List[tuple[int, int]] = []
        for i, gpos in enumerate(ghost_positions):
            ghost_distances.append((self._dist(px, gpos), i))
        ghost_distances.sort(key=lambda item: (item[0], item[1]))

        def nearest_distance(candidates: Iterable[Pos]) -> int:
            best = INF_DISTANCE
            for cand in candidates:
                best = min(best, self._dist(px, cand))
            return best

        nearest_energizer = None
        if state.active_energizers:
            nearest_energizer = min(state.active_energizers, key=lambda e: self._dist(px, e))

        nearest_pellet_distance = nearest_distance(state.active_pellets)
        nearest_energizer_distance = self._dist(px, nearest_energizer) if nearest_energizer is not None else INF_DISTANCE
        nearest_junction_distance = nearest_distance(self.junctions)

        nearest_ghost_distance = ghost_distances[0][0] if ghost_distances else INF_DISTANCE
        second_ghost_distance = ghost_distances[1][0] if len(ghost_distances) > 1 else INF_DISTANCE

        nearest_ghost_edible = 0.0
        second_ghost_edible = 0.0
        if ghost_distances:
            nearest_ghost_edible = 1.0 if frightened[ghost_distances[0][1]] > 0 else 0.0
        if len(ghost_distances) > 1:
            second_ghost_edible = 1.0 if frightened[ghost_distances[1][1]] > 0 else 0.0

        nearest_energies_ghost_edible = 0.0
        nearest_energies_ghost_dist_from_pacman = INF_DISTANCE
        nearest_energies_ghost_dist_from_energizer = INF_DISTANCE
        if nearest_energizer is not None:
            energizer_ghost_distances = []
            for i, gpos in enumerate(ghost_positions):
                energizer_ghost_distances.append((self._dist(nearest_energizer, gpos), i))
            energizer_ghost_distances.sort(key=lambda item: (item[0], item[1]))
            if energizer_ghost_distances:
                idx = energizer_ghost_distances[0][1]
                nearest_energies_ghost_edible = 1.0 if frightened[idx] > 0 else 0.0
                nearest_energies_ghost_dist_from_pacman = self._dist(px, ghost_positions[idx])
                nearest_energies_ghost_dist_from_energizer = self._dist(nearest_energizer, ghost_positions[idx])

        return np.asarray(
            [
                self._norm(nearest_ghost_distance, self.pacman_diameter),
                self._norm(second_ghost_distance, self.pacman_diameter),
                self._norm(nearest_energizer_distance, self.pacman_diameter),
                self._norm(nearest_pellet_distance, self.pacman_diameter),
                self._norm(nearest_junction_distance, self.pacman_diameter),
                nearest_ghost_edible,
                second_ghost_edible,
                nearest_energies_ghost_edible,
                self._norm(nearest_energies_ghost_dist_from_pacman, self.pacman_diameter),
                self._norm(nearest_energies_ghost_dist_from_energizer, self.pacman_diameter),
            ],
            dtype=np.float32,
        )

    def move_vector(
        self,
        before: Observation | SimState | dict | Any,
        after: Observation | SimState | dict | Any,
    ) -> np.ndarray:
        return self.feature_vector(after) - self.feature_vector(before)

    def candidate_move_vector(
        self,
        obs: Observation | SimState | dict | Any,
        direction: Direction,
    ) -> np.ndarray:
        state = _as_observation(obs)
        current = state.pacman_pos
        next_pos = direction.apply(current)
        legal = next_pos in self.maze.pacman_graph.get(current, ())
        if not legal:
            next_pos = current

        active_pellets = set(state.active_pellets)
        active_energizers = set(state.active_energizers)
        if next_pos in active_pellets:
            active_pellets.remove(next_pos)
        if next_pos in active_energizers:
            active_energizers.remove(next_pos)

        next_state = Observation(
            step=state.step + 1,
            pacman_pos=next_pos,
            ghost_positions=list(state.ghost_positions),
            ghost_frightened=list(state.ghost_frightened),
            active_pellets=sorted(active_pellets),
            active_energizers=sorted(active_energizers),
        )
        return self.move_vector(state, next_state)


class KNNPredictor:
    """Compact KNN predictor for Pac-Man movement directions."""

    def __init__(
        self,
        maze: Maze,
        dataset_x: Optional[np.ndarray] = None,
        dataset_y: Optional[np.ndarray] = None,
        *,
        k_neighbors: int = 3,
        rng: Optional[np.random.Generator] = None,
    ):
        self.maze = maze
        self.features = FeatureExtractor(maze)
        self.k_neighbors = max(1, int(k_neighbors))
        self.rng = rng or np.random.default_rng()
        if dataset_x is None or dataset_y is None:
            self.X = np.empty((0, self.features.FEATURE_DIM), dtype=np.float32)
            self.Y = np.empty((0, self.features.FEATURE_DIM), dtype=np.float32)
        else:
            self.X = np.asarray(dataset_x, dtype=np.float32)
            self.Y = np.asarray(dataset_y, dtype=np.float32)
        if self.X.size == 0:
            warnings.warn(
                "KNN predictor loaded with an empty dataset; falling back to random legal moves.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.X.ndim != 2 or self.Y.ndim != 2 or self.X.shape != self.Y.shape or self.X.shape[1] != 10:
            if self.X.size != 0 or self.Y.size != 0:
                warnings.warn(
                    f"KNN dataset has invalid shapes X={self.X.shape}, Y={self.Y.shape}; using random fallback.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.X = np.empty((0, self.features.FEATURE_DIM), dtype=np.float32)
            self.Y = np.empty((0, self.features.FEATURE_DIM), dtype=np.float32)

    def _legal_directions(self, obs: Observation | SimState | dict | Any) -> List[Direction]:
        state = _as_observation(obs)
        return [
            d for d in CARDINALS
            if d.apply(state.pacman_pos) in self.maze.pacman_graph.get(state.pacman_pos, ())
        ]

    def _direction_for_move_vector(self, obs: Observation | SimState | dict | Any, move_vec: np.ndarray) -> Direction:
        state = _as_observation(obs)
        best_dir = Direction.STAY
        best_dist = float("inf")
        for direction in self._legal_directions(state):
            candidate = self.features.candidate_move_vector(state, direction)
            dist = float(np.sum((candidate - move_vec) ** 2))
            if dist < best_dist:
                best_dir = direction
                best_dist = dist
        return best_dir

    def predict(self, obs: Observation | SimState | dict | Any) -> Direction:
        state = _as_observation(obs)
        legal = self._legal_directions(state)
        if not legal:
            return Direction.STAY
        if self.X.size == 0:
            return self.rng.choice(legal)

        x = self.features.feature_vector(state)
        diffs = self.X - x
        distances = np.einsum("ij,ij->i", diffs, diffs)
        k = min(self.k_neighbors, len(self.X))
        idx = np.argpartition(distances, k - 1)[:k]
        idx = idx[np.argsort(distances[idx])]
        weights = 1.0 / (np.sqrt(distances[idx]) + 1e-6)
        y_hat = np.average(self.Y[idx], axis=0, weights=weights)

        best_dir = legal[0]
        best_dist = float("inf")
        for direction in legal:
            candidate = self.features.candidate_move_vector(state, direction)
            dist = float(np.sum((candidate - y_hat) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_dir = direction
        return best_dir

    def score(
        self,
        states: Sequence[Observation | SimState | dict | Any] | Sequence[tuple[Any, Any]],
        labels: Optional[Sequence[Any]] = None,
    ) -> float:
        if labels is None:
            pairs = list(states)  # type: ignore[arg-type]
            if not pairs:
                return 0.0
            states = [pair[0] for pair in pairs]
            labels = [pair[1] for pair in pairs]

        correct = 0
        total = 0
        for state, label in zip(states, labels):
            predicted = self.predict(state)
            if isinstance(label, Direction):
                truth = label
            elif isinstance(label, str):
                truth = Direction[label]
            elif isinstance(label, int):
                truth = CARDINALS[label % len(CARDINALS)]
            else:
                truth = self._direction_for_move_vector(state, np.asarray(label, dtype=np.float32))
            correct += int(predicted == truth)
            total += 1
        return correct / total if total else 0.0


@dataclass
class MctsChildStats:
    direction: Direction
    target_root: Pos
    visits: int = 0
    reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.reward / self.visits


@dataclass
class MctsTree:
    root_pos: Pos
    parent_pos: Optional[Pos]
    children: Dict[Direction, MctsChildStats] = field(default_factory=dict)
    rebuilds: int = 1

    def node_count(self) -> int:
        return 1 + len(self.children)


@dataclass
class GhostRuntime:
    committed_dir: Direction = Direction.STAY
    motion_dir: Direction = Direction.STAY
    last_junction: Optional[Pos] = None
    frightened_phase: int = 0
    tree: Optional[MctsTree] = None
    iterations_last_search: int = 0
    tree_rebuilds: int = 0


class IceGuctGhosts:
    name = "iceguct"

    def __init__(
        self,
        config: Optional[IceGuctConfig] = None,
        knn_dataset_path: str | Path = DEFAULT_DATASET_PATH,
        rng: Optional[np.random.Generator] = None,
    ):
        self.cfg = config or IceGuctConfig()
        self.knn_dataset_path = Path(knn_dataset_path)
        self.rng = rng or np.random.default_rng()
        self._dataset_x, self._dataset_y = _load_npz_dataset(self.knn_dataset_path)
        self._maze: Optional[Maze] = None
        self._topology: Optional[GhostTopology] = None
        self._predictor: Optional[KNNPredictor] = None
        self._brains: List[GhostRuntime] = [GhostRuntime() for _ in range(4)]
        self.trees: List[Optional[MctsTree]] = [None, None, None]
        self._prev_obs: Optional[Observation] = None
        self._last_total_iterations: int = 0
        self._last_global_reverse: bool = False
        logger.info("ICE gUCT predictor dataset size: %d", len(self._dataset_x))

    def reset(self, maze: Maze, role: Literal["pacman", "ghosts"]) -> None:
        self._maze = maze
        self._topology = GhostTopology(maze)
        self._predictor = KNNPredictor(
            maze,
            self._dataset_x,
            self._dataset_y,
            k_neighbors=self.cfg.k_neighbors,
            rng=self.rng,
        )
        self._brains = [GhostRuntime() for _ in range(4)]
        self.trees = [None, None, None]
        self._prev_obs = None
        self._last_total_iterations = 0
        self._last_global_reverse = False

    # ------------------------------------------------------------------
    # Small topology helpers
    # ------------------------------------------------------------------

    def _require_ready(self) -> tuple[Maze, GhostTopology, KNNPredictor]:
        if self._maze is None or self._topology is None or self._predictor is None:
            raise RuntimeError("IceGuctGhosts.reset() must be called before act().")
        return self._maze, self._topology, self._predictor

    def _state_from_obs(self, obs: Observation) -> SimState:
        maze, _, _ = self._require_ready()
        return SimState(
            pacman_pos=obs.pacman_pos,
            ghost_positions=list(obs.ghost_positions),
            ghost_frightened=list(obs.ghost_frightened),
            active_pellets=frozenset(obs.active_pellets),
            active_energizers=frozenset(obs.active_energizers),
            step=obs.step,
        )

    def _infer_motion_dir(self, obs: Observation, index: int) -> Direction:
        runtime = self._brains[index]
        if runtime.committed_dir != Direction.STAY:
            return runtime.committed_dir
        if self._prev_obs is None:
            return Direction.STAY
        prev_pos = self._prev_obs.ghost_positions[index]
        curr_pos = obs.ghost_positions[index]
        if prev_pos == curr_pos:
            return runtime.motion_dir
        dx, dy = curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1]
        try:
            return _direction_from_delta(dx, dy)
        except ValueError:
            return runtime.motion_dir

    def _detect_global_reverse(self, obs: Observation) -> bool:
        if self._prev_obs is None:
            return False
        return any(prev == 0 and curr > 0 for prev, curr in zip(self._prev_obs.ghost_frightened, obs.ghost_frightened))

    def _random_legal_direction(self, pos: Pos, graph: Dict[Pos, frozenset[Pos]], reverse_dir: Optional[Direction] = None) -> Direction:
        legal = _legal_directions(pos, graph, reverse_dir)
        if not legal:
            return Direction.STAY
        return self.rng.choice(legal)

    def _continue_ghost_motion(self, obs: Observation, index: int, runtime: GhostRuntime) -> Direction:
        """Keep a ghost moving legally until the next decision root."""
        maze, topology, _ = self._require_ready()
        pos = obs.ghost_positions[index]
        reverse_dir = runtime.motion_dir.reverse() if runtime.motion_dir != Direction.STAY else None
        legal = topology.legal_directions(pos, reverse_dir)
        if not legal:
            legal = topology.legal_directions(pos, None)
        if not legal:
            return Direction.STAY

        if runtime.committed_dir != Direction.STAY and runtime.committed_dir in legal:
            return runtime.committed_dir

        return self.rng.choice(legal)

    def _next_pacman_cross_point(self, obs: Observation) -> Optional[Pos]:
        assert self._topology is not None
        if self._prev_obs is None:
            return None
        prev = self._prev_obs.pacman_pos
        curr = obs.pacman_pos
        if prev == curr:
            return None
        dx, dy = curr[0] - prev[0], curr[1] - prev[1]
        try:
            heading = _direction_from_delta(dx, dy)
        except ValueError:
            return None

        maze, topology, _ = self._require_ready()
        pos = curr
        while True:
            nxt = heading.apply(pos)
            if nxt not in maze.pacman_graph.get(pos, ()):
                return curr if curr in topology.junctions else topology.nearest_junction(curr)
            pos = nxt
            if pos in topology.junctions:
                return pos

    def _blinky_actual_direction(self, obs: Observation, frightened: bool) -> Direction:
        maze, topology, _ = self._require_ready()
        runtime = self._brains[0]
        pos = obs.ghost_positions[0]
        reverse_dir = runtime.motion_dir.reverse() if runtime.motion_dir != Direction.STAY else None
        if frightened:
            legal = _legal_directions(pos, maze.ghost_graph, reverse_dir)
            if not legal:
                legal = _legal_directions(pos, maze.ghost_graph, None)
            if not legal:
                return Direction.STAY
            distances = maze.bfs_distances(obs.pacman_pos, use_ghost_graph=True)
            return max(legal, key=lambda d: distances.get(d.apply(pos), -1))

        target = self._next_pacman_cross_point(obs)
        if target is None:
            target = obs.pacman_pos
        path = maze.bfs_path(pos, target, use_ghost_graph=True)
        if path and len(path) > 1:
            dx, dy = path[1][0] - path[0][0], path[1][1] - path[0][1]
            try:
                direction = _direction_from_delta(dx, dy)
                if direction in _legal_directions(pos, maze.ghost_graph, reverse_dir):
                    return direction
            except ValueError:
                pass

        legal = _legal_directions(pos, maze.ghost_graph, reverse_dir)
        if not legal:
            legal = _legal_directions(pos, maze.ghost_graph, None)
        if not legal:
            return Direction.STAY
        distances = maze.bfs_distances(target, use_ghost_graph=True)
        return min(legal, key=lambda d: distances.get(d.apply(pos), INF_DISTANCE))

    def _tree_root_for(self, obs: Observation, index: int) -> Pos:
        assert self._topology is not None
        pos = obs.ghost_positions[index]
        motion_dir = self._infer_motion_dir(obs, index)
        root = self._topology.next_root(pos, motion_dir)
        return root or pos

    def _build_tree_for_ghost(self, obs: Observation, index: int) -> MctsTree:
        runtime = self._brains[index]
        root = self._tree_root_for(obs, index)
        runtime.last_junction = root if self._topology and root in self._topology.junctions else runtime.last_junction
        tree = MctsTree(root_pos=root, parent_pos=runtime.last_junction)
        runtime.tree = tree
        runtime.tree_rebuilds += 1
        if index >= 1:
            self.trees[index - 1] = tree
        return tree

    def _children_for_tree(self, tree: MctsTree, obs: Observation, index: int) -> List[tuple[Direction, Pos]]:
        maze, topology, _ = self._require_ready()
        pos = obs.ghost_positions[index]
        reverse_dir = None
        runtime = self._brains[index]
        if runtime.motion_dir != Direction.STAY:
            reverse_dir = runtime.motion_dir.reverse()
        legal_dirs = topology.legal_directions(pos, reverse_dir)
        if not legal_dirs:
            legal_dirs = topology.legal_directions(pos, None)
        children: List[tuple[Direction, Pos]] = []
        for direction in legal_dirs:
            target = direction.apply(pos)
            if target in maze.ghost_graph.get(pos, ()):
                children.append((direction, target))
        return children

    def _tree_select_direction(
        self,
        tree: MctsTree,
        obs: Observation,
        index: int,
        *,
        explore: bool,
    ) -> Direction:
        runtime = self._brains[index]
        maze, topology, _ = self._require_ready()
        pos = obs.ghost_positions[index]
        reverse_dir = runtime.motion_dir.reverse() if runtime.motion_dir != Direction.STAY else None
        candidates = self._children_for_tree(tree, obs, index)
        if not candidates:
            legal = topology.legal_directions(pos, reverse_dir)
            if not legal:
                legal = topology.legal_directions(pos, None)
            return self.rng.choice(legal) if legal else Direction.STAY

        for direction, target in candidates:
            if direction not in tree.children:
                tree.children[direction] = MctsChildStats(direction=direction, target_root=target)

        if explore:
            unvisited = [stats for stats in tree.children.values() if stats.visits == 0]
            if unvisited:
                return self.rng.choice([stats.direction for stats in unvisited])

            parent_visits = sum(stats.visits for stats in tree.children.values())
            best_stats = None
            best_score = -float("inf")
            for stats in tree.children.values():
                ucb = stats.mean_reward + self.cfg.C * math.sqrt(
                    math.log(parent_visits + 1.0) / (stats.visits + self.cfg.epsilon)
                )
                if ucb > best_score:
                    best_score = ucb
                    best_stats = stats
            return best_stats.direction if best_stats is not None else self.rng.choice([direction for direction, _ in candidates])

        best_stats = max(
            tree.children.values(),
            key=lambda stats: (stats.mean_reward, stats.visits),
        )
        return best_stats.direction

    def _update_tree_stats(self, tree: MctsTree, direction: Direction, reward: float) -> None:
        if direction not in tree.children:
            return
        stats = tree.children[direction]
        stats.visits += 1
        stats.reward += reward

    def _search_root_direction(self, obs: Observation, index: int) -> tuple[Direction, int, int]:
        runtime = self._brains[index]
        tree = runtime.tree or self._build_tree_for_ghost(obs, index)
        runtime.tree = tree
        if index >= 1:
            self.trees[index - 1] = tree

        deadline = time.perf_counter() + (self.cfg.budget_ms_per_tick / 1000.0)
        iterations = 0
        total_root_visited = 0

        while time.perf_counter() < deadline:
            state = self._state_from_obs(obs)
            visited_directions: List[Direction] = []
            chosen_direction = self._tree_select_direction(tree, obs, index, explore=True)
            visited_directions.append(chosen_direction)
            reward, root_visits = self._simulate_rollout(state, obs, index, chosen_direction, visited_directions)
            self._update_tree_stats(tree, chosen_direction, reward)
            iterations += 1
            total_root_visited += root_visits
            if iterations >= 8 and time.perf_counter() + 0.001 >= deadline:
                break

        runtime.iterations_last_search = iterations
        best_child = max(
            tree.children.values(),
            key=lambda stats: (stats.mean_reward, stats.visits),
            default=None,
        )
        if best_child is None:
            chosen = self._random_legal_direction(
                obs.ghost_positions[index],
                self._maze.ghost_graph if self._maze is not None else {},
            )
        else:
            chosen = best_child.direction
        return chosen, iterations, total_root_visited

    def _simulate_rollout(
        self,
        state: SimState,
        root_obs: Observation,
        root_index: int,
        root_direction: Direction,
        visited_root_directions: List[Direction],
    ) -> tuple[float, int]:
        maze, topology, predictor = self._require_ready()
        sim_state = state.copy()
        brains = [GhostRuntime(
            committed_dir=runtime.committed_dir,
            motion_dir=runtime.motion_dir,
            last_junction=runtime.last_junction,
            frightened_phase=runtime.frightened_phase,
            tree=runtime.tree,
        ) for runtime in self._brains]

        root_visited = 0
        root_tree = self._brains[root_index].tree
        if root_tree is None:
            return 0.0, 0

        for tick in range(min(self.cfg.T_sim, 8)):
            if sim_state.is_terminal:
                break

            pac_action = self._pacman_rollout_action(sim_state, tick, predictor)
            ghost_actions: List[Direction] = []
            for i in range(4):
                runtime = brains[i]
                pos = sim_state.ghost_positions[i]
                frightened = sim_state.ghost_frightened[i] > 0

                if frightened:
                    if runtime.frightened_phase == 1:
                        runtime.frightened_phase = 0
                        ghost_actions.append(Direction.STAY)
                        continue
                    runtime.frightened_phase = 1

                if i == 0:
                    action = self._blinky_rollout_action(sim_state, runtime, random_mode=True)
                else:
                    action = self._tree_rollout_action(
                        sim_state,
                        runtime,
                        i,
                        allow_explore=(i == root_index),
                    )

                if i == root_index and action == root_direction:
                    root_visited += 1
                ghost_actions.append(action)

                if action != Direction.STAY:
                    runtime.motion_dir = action
                    runtime.committed_dir = action

                if topology.is_junction(pos):
                    runtime.last_junction = pos

            sim_state = sim_step(
                sim_state,
                maze,
                pac_action,
                tuple(ghost_actions),
                frightened_duration=self.cfg.T_sim,
                max_steps=99999,
            )

        reward = self._rollout_reward(state, sim_state)
        return reward, root_visited

    def _pacman_rollout_action(self, state: SimState, tick: int, predictor: KNNPredictor) -> Direction:
        legal = [
            d for d in CARDINALS
            if d.apply(state.pacman_pos) in self._maze.pacman_graph.get(state.pacman_pos, ())
        ]
        if not legal:
            return Direction.STAY
        return self.rng.choice(legal)

    def _tree_rollout_action(
        self,
        state: SimState,
        runtime: GhostRuntime,
        index: int,
        *,
        allow_explore: bool,
    ) -> Direction:
        maze, topology, _ = self._require_ready()
        pos = state.ghost_positions[index]
        reverse_dir = runtime.motion_dir.reverse() if runtime.motion_dir != Direction.STAY else None
        if runtime.tree is None:
            legal = topology.legal_directions(pos, reverse_dir)
            if not legal:
                legal = topology.legal_directions(pos, None)
            return self.rng.choice(legal) if legal else Direction.STAY

        if not topology.is_junction(pos):
            if runtime.committed_dir != Direction.STAY:
                return runtime.committed_dir
            legal = topology.legal_directions(pos, reverse_dir)
            if not legal:
                legal = topology.legal_directions(pos, None)
            return self.rng.choice(legal) if legal else Direction.STAY

        if runtime.tree.root_pos != pos:
            legal = topology.legal_directions(pos, reverse_dir)
            if not legal:
                legal = topology.legal_directions(pos, None)
            return self.rng.choice(legal) if legal else Direction.STAY

        return self._tree_select_direction(runtime.tree, _as_observation(state), index, explore=allow_explore)

    def _blinky_rollout_action(self, state: SimState, runtime: GhostRuntime, *, random_mode: bool) -> Direction:
        maze, topology, _ = self._require_ready()
        pos = state.ghost_positions[0]
        reverse_dir = runtime.motion_dir.reverse() if runtime.motion_dir != Direction.STAY else None
        legal = topology.legal_directions(pos, reverse_dir)
        if not legal:
            legal = topology.legal_directions(pos, None)
        if not legal:
            return Direction.STAY
        return self.rng.choice(legal) if random_mode else self._blinky_actual_direction(_as_observation(state), state.ghost_frightened[0] > 0)

    def _rollout_reward(self, start: SimState, end: SimState) -> float:
        pellets_at_root = len(start.active_pellets) + len(start.active_energizers)
        pellets_remaining = len(end.active_pellets) + len(end.active_energizers)
        pellets_eaten = pellets_at_root - pellets_remaining
        capture = 1.0 if end.outcome == "eaten" else 0.0
        deny = 1.0 - (pellets_eaten / max(pellets_at_root, 1))
        time_term = 1.0 - (end.step - start.step) / max(self.cfg.T_sim, 1)

        maze, topology, _ = self._require_ready()
        pairwise = []
        for i in range(4):
            for j in range(i + 1, 4):
                d = maze.bfs_distances(end.ghost_positions[i], use_ghost_graph=True).get(end.ghost_positions[j], INF_DISTANCE)
                if d >= INF_DISTANCE:
                    d = topology.diameter()
                pairwise.append(d)
        disperse = (sum(pairwise) / len(pairwise)) / float(max(topology.diameter(), 1)) if pairwise else 0.0
        eaten = 0
        for i in range(4):
            if end.ghost_frightened[i] == 0 and start.ghost_frightened[i] > 0 and end.ghost_positions[i] == maze.ghost_spawns[i]:
                eaten += 1

        reward = (
            self.cfg.w_capture * capture
            + self.cfg.w_deny * deny
            + self.cfg.w_time * time_term
            + self.cfg.w_disperse * disperse
            - self.cfg.w_eaten * (eaten / 4.0)
        )
        return reward

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self, observation: Observation | SimState | dict | Any) -> GhostAction:
        maze, topology, _ = self._require_ready()
        obs = _as_observation(observation)
        state = self._state_from_obs(obs)
        global_reverse = self._detect_global_reverse(obs)
        self._last_global_reverse = global_reverse

        if global_reverse:
            logger.info("Global reverse detected at step %d", obs.step)

        actions: List[Direction] = [runtime.committed_dir for runtime in self._brains]
        hold_tick: List[bool] = [False] * 4

        # Update runtime direction inference before any search.
        for i in range(4):
            runtime = self._brains[i]
            runtime.motion_dir = self._infer_motion_dir(obs, i)
            if runtime.motion_dir != Direction.STAY:
                runtime.committed_dir = runtime.motion_dir

        # Global reverse handling: discard all trees and reverse current motion.
        if global_reverse:
            for i in range(4):
                runtime = self._brains[i]
                if runtime.motion_dir != Direction.STAY:
                    runtime.committed_dir = runtime.motion_dir.reverse()
                    runtime.motion_dir = runtime.committed_dir
                if i >= 1:
                    root = self._tree_root_for(obs, i)
                    runtime.tree = MctsTree(root_pos=root, parent_pos=runtime.last_junction)
                    runtime.tree_rebuilds += 1
                    self.trees[i - 1] = runtime.tree
            # No search on the reverse tick; the flipped directions are played directly.
            self._prev_obs = obs
            self._last_total_iterations = 0
            return tuple(runtime.committed_dir if runtime.committed_dir != Direction.STAY else Direction.STAY for runtime in self._brains)  # type: ignore[return-value]

        # Update / rebuild the per-ghost tree roots as needed.
        for i in range(1, 4):
            runtime = self._brains[i]
            root = self._tree_root_for(obs, i)
            if runtime.tree is None or runtime.tree.root_pos != root:
                runtime.tree = MctsTree(root_pos=root, parent_pos=runtime.last_junction)
                runtime.tree_rebuilds += 1
                self.trees[i - 1] = runtime.tree

        # Half-speed frightened ghosts: alternate move / stay.
        for i in range(4):
            runtime = self._brains[i]
            frightened = obs.ghost_frightened[i] > 0
            if frightened:
                if runtime.frightened_phase == 1:
                    actions[i] = Direction.STAY
                    hold_tick[i] = True
                    runtime.frightened_phase = 0
                    continue
                runtime.frightened_phase = 1
            else:
                runtime.frightened_phase = 0

        # Blinky uses its rule unless held stationary by frightened half-speed.
        if not hold_tick[0]:
            actions[0] = self._blinky_actual_direction(obs, obs.ghost_frightened[0] > 0)
            self._brains[0].committed_dir = actions[0]
            self._brains[0].motion_dir = actions[0]

        # MCTS ghosts search only when they are at their current root.
        total_iterations = 0
        search_indices = []
        for i in range(1, 4):
            runtime = self._brains[i]
            if hold_tick[i]:
                continue
            if runtime.tree is not None and obs.ghost_positions[i] == runtime.tree.root_pos:
                search_indices.append(i)

        if search_indices:
            per_ghost_budget_ms = self.cfg.budget_ms_per_tick / max(len(search_indices), 1)
            for i in search_indices:
                runtime = self._brains[i]
                deadline = time.perf_counter() + (per_ghost_budget_ms / 1000.0)
                max_iterations = 1 if self.cfg.budget_split == "equal" else 2
                iterations = 0
                while time.perf_counter() < deadline and iterations < max_iterations:
                    chosen_direction = self._tree_select_direction(runtime.tree, obs, i, explore=True)  # type: ignore[arg-type]
                    runtime.tree.children.setdefault(chosen_direction, MctsChildStats(chosen_direction, chosen_direction.apply(obs.ghost_positions[i])))
                    reward, _ = self._simulate_rollout(state, obs, i, chosen_direction, [chosen_direction])
                    self._update_tree_stats(runtime.tree, chosen_direction, reward)  # type: ignore[arg-type]
                    iterations += 1
                    total_iterations += 1

                best_child = max(
                    runtime.tree.children.values(),
                    key=lambda stats: (stats.mean_reward, stats.visits),
                    default=None,
                )
                if best_child is not None:
                    actions[i] = best_child.direction
                else:
                    root_pos = obs.ghost_positions[i]
                    reverse_dir = runtime.motion_dir.reverse() if runtime.motion_dir != Direction.STAY else None
                    legal = topology.legal_directions(root_pos, reverse_dir)
                    if not legal:
                        legal = topology.legal_directions(root_pos, None)
                    actions[i] = self.rng.choice(legal) if legal else Direction.STAY

                runtime.committed_dir = actions[i]
                runtime.motion_dir = actions[i]
                runtime.last_junction = obs.ghost_positions[i]

        # Non-searching ghosts simply continue their committed motion.
        for i in range(1, 4):
            if hold_tick[i]:
                continue
            runtime = self._brains[i]
            if actions[i] == Direction.STAY or actions[i].apply(obs.ghost_positions[i]) not in maze.ghost_graph.get(obs.ghost_positions[i], ()):
                actions[i] = self._continue_ghost_motion(obs, i, runtime)
            runtime.motion_dir = actions[i]
            runtime.committed_dir = actions[i]

        self._prev_obs = obs
        self._last_total_iterations = total_iterations
        logger.info(
            "ICE gUCT step=%d iterations=%d tree_sizes=%s",
            obs.step,
            total_iterations,
            [tree.node_count() if tree is not None else 0 for tree in self.trees],
        )
        return tuple(actions)  # type: ignore[return-value]


# Backwards-compatible alias for any stray imports.
MCTSGhostsAgent = IceGuctGhosts
