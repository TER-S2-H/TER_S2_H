#!/usr/bin/env python3
"""
Pac-Man session simulator for JEPA data capture.

Runs autonomous Pac-Man vs 4 ghosts sessions from a maze JSON and stores
one self-contained pickle per session.
"""

import argparse
import json
import os
import pickle
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

Coord = Tuple[int, int]


def parse_coord_string(value: str) -> Coord:
    x_str, y_str = value.split(",")
    return int(x_str), int(y_str)


def infer_seed_from_maze_path(path: str) -> Optional[str]:
    match = re.search(r"maze_(\d+)", path)
    if match:
        return match.group(1)
    return None


def load_maze(path: str):
    """Parse maze JSON and return core components.

    Supports both edge schemas:
    - [{"from": "x,y", "to": "x,y"}]
    - [{"x1": x, "y1": y, "x2": x, "y2": y}]

    Pellets are derived from PATH nodes when the JSON has no explicit
    "pellets" field.
    """
    with open(path, "r") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    ghost_walls = {(w["x"], w["y"]) for w in data.get("ghost_walls", [])}

    energizers = {(e["x"], e["y"]) for e in data.get("energizers", [])}

    if "pellets" in data:
        pellets = {(p["x"], p["y"]) for p in data.get("pellets", [])}
    else:
        pellets = {(n["x"], n["y"]) for n in nodes if n.get("type") == "PATH"}
        pellets -= energizers

    metadata = data.get("metadata", {})

    return {
        "metadata": metadata,
        "nodes": nodes,
        "edges": edges,
        "pellets": pellets,
        "energizers": energizers,
        "ghost_walls": ghost_walls,
    }


def symmetrize_energizers(
    energizers: Set[Coord],
    node_to_index: Dict[Coord, int],
    height: int,
) -> Set[Coord]:
    """Extend energizers to 4 positions by vertical mirroring and nearest fallback."""
    node_coords = list(node_to_index.keys())
    if not node_coords:
        return set()

    min_x = min(x for x, _ in node_coords)
    max_x = max(x for x, _ in node_coords)
    min_y = min(y for _, y in node_coords)
    max_y = max(y for _, y in node_coords)

    def nearest_node(target: Coord, excluded: Set[Coord]) -> Coord:
        candidates = [c for c in node_coords if c not in excluded]
        if not candidates:
            candidates = node_coords
        return min(
            candidates,
            key=lambda c: ((c[0] - target[0]) ** 2 + (c[1] - target[1]) ** 2, c[0], c[1]),
        )

    base = sorted(energizers)
    full = set(base)

    for x, y in base:
        mirror = (x, height - 1 - y)
        if mirror in node_to_index:
            full.add(mirror)
        else:
            full.add(nearest_node(mirror, full))

    corner_targets = [
        (min_x, min_y),
        (max_x, min_y),
        (min_x, max_y),
        (max_x, max_y),
    ]

    for target in corner_targets:
        if len(full) >= 4:
            break
        full.add(nearest_node(target, full))

    if len(full) > 4:
        full = set(
            sorted(
                full,
                key=lambda c: min(
                    (c[0] - tx) ** 2 + (c[1] - ty) ** 2 for tx, ty in corner_targets
                ),
            )[:4]
        )

    return full


def build_graph(
    nodes: List[dict],
    edges: List[dict],
    width: Optional[int] = None,
    height: Optional[int] = None,
):
    """Build graph structures including COO, mappings, and adjacency."""
    node_coords = [(n["x"], n["y"]) for n in nodes]

    node_to_index: Dict[Coord, int] = {coord: idx for idx, coord in enumerate(node_coords)}
    index_to_coord: Dict[int, Coord] = {idx: coord for coord, idx in node_to_index.items()}

    coo_src: List[int] = []
    coo_dst: List[int] = []
    adjacency: Dict[int, List[int]] = {i: [] for i in range(len(node_coords))}

    def add_directed(a_idx: int, b_idx: int):
        coo_src.append(a_idx)
        coo_dst.append(b_idx)
        adjacency[a_idx].append(b_idx)

    for edge in edges:
        if "from" in edge and "to" in edge:
            a = parse_coord_string(edge["from"])
            b = parse_coord_string(edge["to"])
        else:
            a = (edge["x1"], edge["y1"])
            b = (edge["x2"], edge["y2"])

        if a not in node_to_index or b not in node_to_index:
            continue

        ai = node_to_index[a]
        bi = node_to_index[b]

        add_directed(ai, bi)
        add_directed(bi, ai)

    for idx in adjacency:
        adjacency[idx] = sorted(set(adjacency[idx]))

    # Add wrap-around tunnel edges (left <-> right) when both boundary nodes exist.
    if width is not None and height is not None and width > 1:
        for y in range(height):
            left = (0, y)
            right = (width - 1, y)
            if left in node_to_index and right in node_to_index:
                li = node_to_index[left]
                ri = node_to_index[right]

                if ri not in adjacency[li]:
                    coo_src.append(li)
                    coo_dst.append(ri)
                    adjacency[li].append(ri)

                if li not in adjacency[ri]:
                    coo_src.append(ri)
                    coo_dst.append(li)
                    adjacency[ri].append(li)

        for idx in adjacency:
            adjacency[idx] = sorted(set(adjacency[idx]))

    return coo_src, coo_dst, node_to_index, index_to_coord, adjacency


def compute_visibility(
    agent_node: int,
    adjacency: Dict[int, List[int]],
    index_to_coord: Dict[int, Coord],
    node_to_index: Dict[Coord, int],
) -> Set[int]:
    """Compute line-of-sight visibility in 4 cardinal directions."""
    visible: Set[int] = set()
    x0, y0 = index_to_coord[agent_node]

    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        cur_idx = agent_node
        x, y = x0, y0

        while True:
            nx, ny = x + dx, y + dy
            nxt = node_to_index.get((nx, ny))
            if nxt is None:
                break
            if nxt not in adjacency[cur_idx]:
                break

            visible.add(nxt)
            cur_idx = nxt
            x, y = nx, ny

    visible.discard(agent_node)
    return visible


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearest_coord_to(target: Coord, candidates: List[Coord]) -> Coord:
    return min(candidates, key=lambda c: ((c[0] - target[0]) ** 2 + (c[1] - target[1]) ** 2, c[0], c[1]))


def choose_best_neighbor(
    candidates: List[int],
    index_to_coord: Dict[int, Coord],
    target_coords: List[Coord],
    maximize: bool,
) -> int:
    if not target_coords:
        return random.choice(candidates)

    scored: List[Tuple[int, int]] = []
    for cand in candidates:
        c = index_to_coord[cand]
        best = min(manhattan(c, t) for t in target_coords)
        scored.append((cand, best))

    if maximize:
        best_value = max(score for _, score in scored)
    else:
        best_value = min(score for _, score in scored)

    best_nodes = [node for node, score in scored if score == best_value]
    return random.choice(best_nodes)


def choose_flee_neighbor_strict(
    current_node: int,
    candidates: List[int],
    index_to_coord: Dict[int, Coord],
    threat_nodes: List[int],
) -> int:
    """Choose a flee move that increases safety when possible.

    Safety metric is min Manhattan distance to all threat nodes.
    If any candidate strictly improves over current safety, only those are allowed.
    """
    current_coord = index_to_coord[current_node]
    threat_coords = [index_to_coord[t] for t in threat_nodes]

    current_safety = min(manhattan(current_coord, t) for t in threat_coords)
    scored = []
    for cand in candidates:
        c = index_to_coord[cand]
        safety = min(manhattan(c, t) for t in threat_coords)
        scored.append((cand, safety))

    improving = [cand for cand, safety in scored if safety > current_safety]
    if improving:
        best_value = max(safety for cand, safety in scored if cand in improving)
        best_nodes = [cand for cand, safety in scored if cand in improving and safety == best_value]
        return random.choice(best_nodes)

    # If no strictly better move exists, keep previous maximize behavior.
    best_value = max(safety for _, safety in scored)
    best_nodes = [cand for cand, safety in scored if safety == best_value]
    return random.choice(best_nodes)


def get_candidates(
    agent_node: int,
    previous_node: Optional[int],
    adjacency: Dict[int, List[int]],
    blocked_nodes: Optional[Set[int]] = None,
) -> List[int]:
    """Get move candidates using reversal rule, then blocked-node exclusion."""
    neighbors = list(adjacency[agent_node])
    if previous_node is not None and previous_node in neighbors and len(neighbors) > 1:
        neighbors = [n for n in neighbors if n != previous_node]

    if blocked_nodes:
        neighbors = [n for n in neighbors if n not in blocked_nodes]

    return neighbors


def is_dangerous_ghost(state: "GameState", ghost_idx: int) -> bool:
    """A ghost is dangerous only when not frightened."""
    return state.ghost_frightened[ghost_idx] == 0 and not state.ghost_in_house[ghost_idx]


def default_house_release_paths() -> List[List[Coord]]:
    """Paths from inside ghost-house void to top entry point.

    Final coordinate of each path must be a graph node where the ghost enters.
    """
    return [
        [(12, 14), (13, 14), (13, 13), (13, 12), (13, 11)],
        [(13, 14), (14, 14), (14, 13), (14, 12), (14, 11)],
        [(14, 14), (13, 14), (13, 13), (13, 12), (13, 11)],
        [(15, 14), (14, 14), (14, 13), (14, 12), (14, 11)],
    ]


def extract_walkable_and_spawns(
    nodes: List[dict],
    node_to_index: Dict[Coord, int],
    ghost_walls: Set[Coord],
):
    path_coords = [(n["x"], n["y"]) for n in nodes if n.get("type") == "PATH"]
    pathblank_coords = [(n["x"], n["y"]) for n in nodes if n.get("type") == "PATHBLANK"]
    all_coords = [(n["x"], n["y"]) for n in nodes]

    # Prefer exact classic spawn coordinate when available.
    if (13, 23) in node_to_index:
        pacman_spawn_coord = (13, 23)
    else:
        pacman_spawn_coord = nearest_coord_to((13, 23), path_coords if path_coords else all_coords)

    center = (13.5, 14.0)

    def center_dist(c: Coord) -> float:
        return (c[0] - center[0]) ** 2 + (c[1] - center[1]) ** 2

    ghost_spawn_coords: List[Coord] = []

    # Prefer in-house spawns below the gate (not on the entry row).
    preferred_house_spawns = [(11, 17), (12, 17), (15, 17), (16, 17)]
    for c in preferred_house_spawns:
        if c in node_to_index and c in set(pathblank_coords):
            ghost_spawn_coords.append(c)

    if len(ghost_spawn_coords) == 4:
        pacman_spawn = node_to_index[pacman_spawn_coord]
        ghost_spawns = [node_to_index[c] for c in ghost_spawn_coords]
        return pacman_spawn, ghost_spawns

    if ghost_walls and pathblank_coords:
        gate_center_x = sum(x for x, _ in ghost_walls) / len(ghost_walls)
        gate_center_y = sum(y for _, y in ghost_walls) / len(ghost_walls)

        # Inside-house candidates are below the gate line.
        inside_house = [c for c in pathblank_coords if c[1] > gate_center_y]
        if len(inside_house) >= 4:
            inside_house = sorted(
                set(inside_house),
                key=lambda c: (
                    (c[0] - gate_center_x) ** 2 + (c[1] - (gate_center_y + 5)) ** 2,
                    c[0],
                    c[1],
                ),
            )
            ghost_spawn_coords = inside_house[:4]
            pacman_spawn = node_to_index[pacman_spawn_coord]
            ghost_spawns = [node_to_index[c] for c in ghost_spawn_coords]
            return pacman_spawn, ghost_spawns

        def gate_dist(c: Coord) -> int:
            return min(abs(c[0] - gx) + abs(c[1] - gy) for gx, gy in ghost_walls)

        radius = 1
        near_gate: List[Coord] = []
        while radius <= 10:
            near_gate = [c for c in pathblank_coords if gate_dist(c) <= radius and c[1] > gate_center_y]
            if len(near_gate) >= 4:
                break
            radius += 1

        near_gate = sorted(
            set(near_gate),
            key=lambda c: (
                gate_dist(c),
                (c[0] - gate_center_x) ** 2 + (c[1] - gate_center_y) ** 2,
                c[0],
                c[1],
            ),
        )
        ghost_spawn_coords = near_gate[:4]

    if len(ghost_spawn_coords) < 4:
        house_candidates = [
            c for c in pathblank_coords if 11 <= c[0] <= 16 and 13 <= c[1] <= 15
        ]
        house_candidates = sorted(set(house_candidates), key=lambda c: (center_dist(c), c[0], c[1]))
        for c in house_candidates:
            if c not in ghost_spawn_coords:
                ghost_spawn_coords.append(c)
            if len(ghost_spawn_coords) == 4:
                break

    if len(ghost_spawn_coords) < 4:
        fallback = sorted(set(pathblank_coords), key=lambda c: (center_dist(c), c[0], c[1]))
        for c in fallback:
            if c not in ghost_spawn_coords:
                ghost_spawn_coords.append(c)
            if len(ghost_spawn_coords) == 4:
                break

    if len(ghost_spawn_coords) < 4:
        fallback = sorted(set(all_coords), key=lambda c: (center_dist(c), c[0], c[1]))
        for c in fallback:
            if c not in ghost_spawn_coords:
                ghost_spawn_coords.append(c)
            if len(ghost_spawn_coords) == 4:
                break

    pacman_spawn = node_to_index[pacman_spawn_coord]
    ghost_spawns = [node_to_index[c] for c in ghost_spawn_coords]

    return pacman_spawn, ghost_spawns


@dataclass
class GameState:
    adjacency: Dict[int, List[int]]
    node_to_index: Dict[Coord, int]
    index_to_coord: Dict[int, Coord]
    ghost_wall_nodes: Set[int]
    pacman_spawn: int
    ghost_spawns: List[int]
    initial_pellets: Set[int]
    initial_energizers: Set[int]
    max_steps: int = 3000

    pacman_pos: int = field(init=False)
    pacman_prev: Optional[int] = field(init=False, default=None)

    ghost_positions: List[int] = field(init=False)
    ghost_prev: List[Optional[int]] = field(init=False)
    ghost_frightened: List[int] = field(init=False)
    ghost_in_house: List[bool] = field(init=False)
    ghost_house_paths: List[List[Coord]] = field(init=False)
    ghost_house_progress: List[int] = field(init=False)

    active_pellets: Set[int] = field(init=False)
    active_energizers: Set[int] = field(init=False)

    step_count: int = field(init=False, default=0)
    frames: List[dict] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.pacman_pos = self.pacman_spawn
        self.ghost_positions = list(self.ghost_spawns)
        self.ghost_prev = [None, None, None, None]
        self.ghost_frightened = [0, 0, 0, 0]
        self.ghost_in_house = [True, True, True, True]
        self.ghost_house_paths = default_house_release_paths()
        self.ghost_house_progress = [0, 0, 0, 0]

        # If a release endpoint does not exist in graph, disable house mode for that ghost.
        for i in range(4):
            end_coord = self.ghost_house_paths[i][-1]
            if end_coord in self.node_to_index:
                self.ghost_positions[i] = self.node_to_index[end_coord]
            else:
                self.ghost_in_house[i] = False

        self.active_pellets = set(self.initial_pellets)
        self.active_energizers = set(self.initial_energizers)

        self.record_frame(step=0)

    def record_frame(self, step: int):
        ghost_coords = []
        for i in range(4):
            if self.ghost_in_house[i]:
                p = self.ghost_house_progress[i]
                ghost_coords.append(self.ghost_house_paths[i][p])
            else:
                ghost_coords.append(self.index_to_coord[self.ghost_positions[i]])

        self.frames.append(
            {
                "step": step,
                "pacman_pos": self.index_to_coord[self.pacman_pos],
                "ghost_positions": ghost_coords,
                "ghost_frightened": list(self.ghost_frightened),
                "active_pellets": sorted(self.index_to_coord[p] for p in self.active_pellets),
                "active_energizers": sorted(self.index_to_coord[e] for e in self.active_energizers),
            }
        )


def choose_pacman_move(state: GameState, vis_pac: Set[int], is_first_step: bool = False) -> int:
    all_neighbors = [
        n for n in state.adjacency[state.pacman_pos] if n not in state.ghost_wall_nodes
    ]

    candidates = get_candidates(
        state.pacman_pos,
        state.pacman_prev,
        state.adjacency,
        blocked_nodes=state.ghost_wall_nodes,
    )
    if not candidates:
        return state.pacman_pos

    visible_non_frightened_ghosts = [
        state.ghost_positions[i]
        for i in range(4)
        if is_dangerous_ghost(state, i) and state.ghost_positions[i] in vis_pac
    ]

    visible_frightened_ghosts = [
        i
        for i in range(4)
        if (not is_dangerous_ghost(state, i)) and state.ghost_positions[i] in vis_pac
    ]

    def maybe_allow_emergency_reverse(threat_nodes: List[int]) -> List[int]:
        if state.pacman_prev is None or state.pacman_prev not in all_neighbors:
            return candidates
        if state.pacman_prev in state.ghost_wall_nodes:
            return candidates

        threat_coords = [state.index_to_coord[t] for t in threat_nodes]
        reverse_coord = state.index_to_coord[state.pacman_prev]
        reverse_safety = min(manhattan(reverse_coord, t) for t in threat_coords)

        if candidates:
            best_forward_safety = max(
                min(manhattan(state.index_to_coord[c], t) for t in threat_coords)
                for c in candidates
            )
        else:
            best_forward_safety = -1

        if reverse_safety > best_forward_safety:
            return candidates + [state.pacman_prev]
        return candidates

    if visible_non_frightened_ghosts:
        flee_candidates = maybe_allow_emergency_reverse(visible_non_frightened_ghosts)
        return choose_flee_neighbor_strict(
            state.pacman_pos,
            flee_candidates,
            state.index_to_coord,
            visible_non_frightened_ghosts,
        )

    if visible_frightened_ghosts:
        cur_coord = state.index_to_coord[state.pacman_pos]
        pellet_candidates = [
            n for n in vis_pac if n in state.active_pellets or n in state.active_energizers
        ]
        nearest_pellet_distance = None
        if pellet_candidates:
            nearest_pellet_distance = min(
                manhattan(cur_coord, state.index_to_coord[n]) for n in pellet_candidates
            )

        chaseable_targets: List[Tuple[int, int]] = []  # (ghost_idx, distance)
        for ghost_idx in visible_frightened_ghosts:
            ghost_coord = state.index_to_coord[state.ghost_positions[ghost_idx]]
            distance = manhattan(cur_coord, ghost_coord)
            # Only chase if the frightened timer still gives us a realistic chance.
            if state.ghost_frightened[ghost_idx] > distance:
                if nearest_pellet_distance is None or distance < nearest_pellet_distance:
                    chaseable_targets.append((ghost_idx, distance))

        if chaseable_targets:
            best_distance = min(distance for _, distance in chaseable_targets)
            best_ghosts = [
                ghost_idx for ghost_idx, distance in chaseable_targets if distance == best_distance
            ]
            target_ghost_idx = random.choice(best_ghosts)
            target = [state.index_to_coord[state.ghost_positions[target_ghost_idx]]]
            return choose_best_neighbor(candidates, state.index_to_coord, target, maximize=False)

    visible_collectibles = [
        n for n in vis_pac if n in state.active_pellets or n in state.active_energizers
    ]
    if visible_collectibles:
        cur_coord = state.index_to_coord[state.pacman_pos]
        collectible_coords = [state.index_to_coord[n] for n in visible_collectibles]
        min_dist = min(manhattan(cur_coord, c) for c in collectible_coords)
        nearest_targets = [c for c in collectible_coords if manhattan(cur_coord, c) == min_dist]
        target = random.choice(nearest_targets)
        return choose_best_neighbor(candidates, state.index_to_coord, [target], maximize=False)

    return random.choice(candidates)


def choose_ghost_move(state: GameState, ghost_idx: int, vis_ghost: Set[int]) -> int:
    if state.ghost_in_house[ghost_idx]:
        return state.ghost_positions[ghost_idx]

    pos = state.ghost_positions[ghost_idx]
    prev = state.ghost_prev[ghost_idx]
    all_neighbors = list(state.adjacency[pos])

    candidates = get_candidates(pos, prev, state.adjacency, blocked_nodes=None)
    if not candidates:
        candidates = all_neighbors if all_neighbors else [pos]

    pac_visible = state.pacman_pos in vis_ghost

    def maybe_allow_emergency_reverse_for_flee() -> List[int]:
        if prev is None or prev not in all_neighbors:
            return candidates
        current_coord = state.index_to_coord[pos]
        pac_coord = state.index_to_coord[state.pacman_pos]
        reverse_safety = manhattan(state.index_to_coord[prev], pac_coord)

        if candidates:
            best_forward_safety = max(
                manhattan(state.index_to_coord[c], pac_coord) for c in candidates
            )
        else:
            best_forward_safety = -1

        if reverse_safety > best_forward_safety:
            return candidates + [prev]
        return candidates

    if state.ghost_frightened[ghost_idx] > 0:
        if pac_visible:
            flee_candidates = maybe_allow_emergency_reverse_for_flee()
            return choose_flee_neighbor_strict(
                pos,
                flee_candidates,
                state.index_to_coord,
                [state.pacman_pos],
            )
        return random.choice(candidates)

    if pac_visible:
        target = [state.index_to_coord[state.pacman_pos]]
        return choose_best_neighbor(candidates, state.index_to_coord, target, maximize=False)

    return random.choice(candidates)


def check_collision(state: GameState) -> bool:
    """Return True if Pac-Man is on a non-frightened ghost node."""
    return any(
        state.ghost_positions[i] == state.pacman_pos and is_dangerous_ghost(state, i)
        for i in range(4)
    )


def check_crossing(
    old_pacman: int,
    new_pacman: int,
    old_ghosts: List[int],
    new_ghosts: List[int],
    frightened: List[int],
    old_in_house: List[bool],
    new_in_house: List[bool],
) -> bool:
    """Return True if Pac-Man and a non-frightened ghost swapped nodes."""
    for i in range(4):
        if old_in_house[i] or new_in_house[i]:
            continue
        if frightened[i] == 0 and new_pacman == old_ghosts[i] and new_ghosts[i] == old_pacman:
            return True
    return False


def step(state: GameState, is_first_step: bool):
    """Execute one full simultaneous step and return (done, outcome)."""
    # 1. Pre-move collision check.
    if check_collision(state):
        return True, "eaten"

    # 2. Compute visibility.
    vis_pac = compute_visibility(
        state.pacman_pos,
        state.adjacency,
        state.index_to_coord,
        state.node_to_index,
    )
    vis_ghosts = []
    for i, g in enumerate(state.ghost_positions):
        if state.ghost_in_house[i]:
            vis_ghosts.append(set())
        else:
            vis_ghosts.append(
                compute_visibility(g, state.adjacency, state.index_to_coord, state.node_to_index)
            )

    # 3. Compute intended moves in parallel.
    next_pacman = choose_pacman_move(state, vis_pac, is_first_step=is_first_step)

    next_ghosts = [choose_ghost_move(state, i, vis_ghosts[i]) for i in range(4)]

    old_in_house = list(state.ghost_in_house)

    # 4. Save old positions (as previous nodes for next-step reversal rule).
    old_pacman = state.pacman_pos
    old_ghosts = list(state.ghost_positions)

    state.pacman_prev = state.pacman_pos
    state.ghost_prev = list(state.ghost_positions)

    # 5. Apply moves simultaneously.
    state.pacman_pos = next_pacman
    state.ghost_positions = next_ghosts

    # Progress in-house ghosts toward the entry and activate them once they reach endpoint.
    for i in range(4):
        if not state.ghost_in_house[i]:
            continue
        if state.ghost_house_progress[i] < len(state.ghost_house_paths[i]) - 1:
            state.ghost_house_progress[i] += 1
        if state.ghost_house_progress[i] == len(state.ghost_house_paths[i]) - 1:
            end_coord = state.ghost_house_paths[i][-1]
            if end_coord in state.node_to_index:
                state.ghost_positions[i] = state.node_to_index[end_coord]
                state.ghost_prev[i] = None
                state.ghost_in_house[i] = False

    new_in_house = list(state.ghost_in_house)

    state.step_count += 1

    # 6. Crossing collision check.
    if check_crossing(
        old_pacman,
        state.pacman_pos,
        old_ghosts,
        state.ghost_positions,
        state.ghost_frightened,
        old_in_house,
        new_in_house,
    ):
        state.record_frame(step=state.step_count)
        return True, "eaten"

    # 7. Resolve events in order.
    if state.pacman_pos in state.active_pellets:
        state.active_pellets.remove(state.pacman_pos)

    if state.pacman_pos in state.active_energizers:
        state.active_energizers.remove(state.pacman_pos)
        state.ghost_frightened = [10, 10, 10, 10]

    if check_collision(state):
        state.record_frame(step=state.step_count)
        return True, "eaten"

    for i in range(4):
        if state.ghost_positions[i] == state.pacman_pos and state.ghost_frightened[i] > 0:
            state.ghost_positions[i] = state.ghost_spawns[i]
            state.ghost_prev[i] = None
            state.ghost_frightened[i] = 0
            state.ghost_in_house[i] = True
            state.ghost_house_progress[i] = 0

    for i in range(4):
        if state.ghost_frightened[i] > 0:
            state.ghost_frightened[i] -= 1

    # 8/9. Termination checks.
    if not state.active_pellets and not state.active_energizers:
        state.record_frame(step=state.step_count)
        return True, "win"
    elif state.step_count >= state.max_steps:
        state.record_frame(step=state.step_count)
        return True, "timeout"

    # 10. Record frame.
    state.record_frame(step=state.step_count)
    return False, None


def run_session(
    graph: dict,
    spawns: dict,
    initial_pellets: Set[int],
    initial_energizers: Set[int],
    session_id: int,
    max_steps: int = 3000,
):
    """Run a full session and return the serialized session dictionary."""
    random.seed(session_id)

    state = GameState(
        adjacency=graph["adjacency"],
        node_to_index=graph["node_to_index"],
        index_to_coord=graph["index_to_coord"],
        ghost_wall_nodes=set(graph["ghost_wall_nodes"]),
        pacman_spawn=spawns["pacman_spawn"],
        ghost_spawns=spawns["ghost_spawns"],
        initial_pellets=set(initial_pellets),
        initial_energizers=set(initial_energizers),
        max_steps=max_steps,
    )

    outcome = "timeout"
    while True:
        done, maybe_outcome = step(state, is_first_step=(state.step_count == 0))
        if done:
            outcome = maybe_outcome or "timeout"
            break

    session = {
        "graph": {
            "coo_src": graph["coo_src"],
            "coo_dst": graph["coo_dst"],
            "node_to_index": graph["node_to_index"],
            "index_to_coord": graph["index_to_coord"],
            "ghost_wall_nodes": sorted(graph["ghost_wall_nodes"]),
            "num_nodes": graph["num_nodes"],
        },
        "initial_pellets": sorted(graph["index_to_coord"][p] for p in initial_pellets),
        "initial_energizers": sorted(graph["index_to_coord"][e] for e in initial_energizers),
        "pacman_spawn": graph["index_to_coord"][spawns["pacman_spawn"]],
        "ghost_spawns": [graph["index_to_coord"][g] for g in spawns["ghost_spawns"]],
        "outcome": outcome,
        "total_steps": state.step_count,
        "frames": state.frames,
    }

    return session


def main():
    parser = argparse.ArgumentParser(description="Pac-Man 1000-session simulator")
    parser.add_argument("maze_json", help="Path to maze.json")
    parser.add_argument("-o", "--output-dir", default="./sessions", help="Base output directory")
    parser.add_argument("-n", "--num-sessions", type=int, default=1000, help="Number of sessions")
    parser.add_argument("--max-steps", type=int, default=3000, help="Session timeout in steps")
    parser.add_argument("--seed", help="Maze seed used to name output subfolder")
    args = parser.parse_args()

    maze = load_maze(args.maze_json)

    meta = maze.get("metadata", {})
    coo_src, coo_dst, node_to_index, index_to_coord, adjacency = build_graph(
        maze["nodes"],
        maze["edges"],
        width=int(meta.get("width", 0)) if meta.get("width") is not None else None,
        height=int(meta.get("height", 0)) if meta.get("height") is not None else None,
    )

    ghost_wall_nodes = {
        node_to_index[c] for c in maze["ghost_walls"] if c in node_to_index
    }

    pacman_spawn, ghost_spawns = extract_walkable_and_spawns(
        maze["nodes"],
        node_to_index,
        maze["ghost_walls"],
    )

    sym_energizers = symmetrize_energizers(
        set(maze["energizers"]),
        node_to_index,
        int(maze["metadata"].get("height", 31)),
    )

    initial_pellets = {
        node_to_index[c] for c in maze["pellets"] if c in node_to_index
    }
    initial_energizers = {
        node_to_index[c] for c in sym_energizers if c in node_to_index
    }

    # Energizer nodes are not pellets.
    initial_pellets -= initial_energizers

    graph = {
        "coo_src": coo_src,
        "coo_dst": coo_dst,
        "node_to_index": node_to_index,
        "index_to_coord": index_to_coord,
        "adjacency": adjacency,
        "ghost_wall_nodes": ghost_wall_nodes,
        "num_nodes": len(node_to_index),
    }

    spawns = {
        "pacman_spawn": pacman_spawn,
        "ghost_spawns": ghost_spawns,
    }

    seed_name = args.seed if args.seed else infer_seed_from_maze_path(args.maze_json)
    if seed_name is None:
        seed_name = "unknown_seed"

    target_output_dir = os.path.join(args.output_dir, str(seed_name))
    os.makedirs(target_output_dir, exist_ok=True)

    for session_id in range(args.num_sessions):
        session = run_session(
            graph,
            spawns,
            initial_pellets,
            initial_energizers,
            session_id=session_id,
            max_steps=args.max_steps,
        )

        out_file = os.path.join(target_output_dir, f"session_{session_id:04d}.pkl")
        with open(out_file, "wb") as f:
            pickle.dump(session, f)

        if session_id % 100 == 0:
            print(
                f"Completed session {session_id:04d} | "
                f"outcome: {session['outcome']} | steps: {session['total_steps']}"
            )


if __name__ == "__main__":
    main()
