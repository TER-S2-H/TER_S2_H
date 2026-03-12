#!/usr/bin/env python3
"""
Compute a maze complexity score: the minimum steps to visit all food tiles
starting from a fixed position (OPEN walk — no return needed).

Problem type: Metric TSP Hamiltonian Path (NP-hard in general).
  We want the ordering of all food-tile nodes that minimises the total
  sum of BFS shortest-path distances between consecutive nodes.

Algorithm:
  For <= 20 reachable nodes: exact Held-Karp DP (provably optimal).
  For larger mazes, a multi-start local-search pipeline:
    1. k greedy nearest-neighbour starts (first deterministic, rest randomised).
    2. Each is refined with:
         - Deterministic 2-opt scan (all pairs, repeat until convergence)
         - Or-opt for chain lengths 1, 2, 3 (relocate + reverse, best-improvement)
         - Another 2-opt scan (Or-opt can open new 2-opt gains)
         - Random 3-opt segment-swap moves (escape deeper local optima)
    3. The best result across all starts is returned.

Lower bound:
  The MST weight is a valid lower bound: true optimum >= MST_weight.
  True optimum lies in [MST_weight, best_found].

Score formula:
  score = (food_count / steps) * 10   (capped at 10, always in [0, 10])
"""

import argparse
import json
import math
import random as _random
from collections import deque
from typing import Deque, Dict, List, Set, Tuple

Position = Tuple[int, int]

# ---------------------------------------------------------------------------
# Module-level constants for score normalisation and weighting
# ---------------------------------------------------------------------------

# Minimum interior length for a corridor to be counted as a trap
TRAP_MIN_INTERIOR: int = 5

# Mean and std of each raw signal — derived from 500 generated mazes (28×31)
# using the greedy path algorithm (seed=42).
# Each signal is z-normalised then squashed: s = tanh((raw - MEAN) / STD)
# Signals live in (-1, +1); negative = easier than average, positive = harder.
#
# The three signals are nearly uncorrelated (max |r| = 0.21 across 500 mazes),
# so equal weighting 1/3 is well-justified. s_turns was dropped because it
# overlaps heavily with s_steps (r = +0.56).
MEAN_STEPS_RATIO: float = 1.28347   # mean of steps / food_count
STD_STEPS_RATIO:  float = 0.04925   # std  of steps / food_count
MEAN_TRAP_LEN:    float = 16.27000  # mean of longest_trap (interior > TRAP_MIN_INTERIOR)
STD_TRAP_LEN:     float = 2.81089   # std  of longest_trap
MEAN_TRAP_CNT:    float = 18.48800  # mean of num_traps
STD_TRAP_CNT:     float = 3.05317   # std  of num_traps

# Combination weights — must sum to 1.0
W_STEPS:    float = 1 / 3
W_TRAP_LEN: float = 1 / 3
W_TRAP_CNT: float = 1 / 3

assert abs(W_STEPS + W_TRAP_LEN + W_TRAP_CNT - 1.0) < 1e-9, (
    "Score weights must sum to 1.0"
)


def _parse_node_id(node_id: str) -> Position:
    x_str, y_str = node_id.split(",")
    return int(x_str), int(y_str)


def _build_graph(maze_data: Dict, wrap_tunnels: bool) -> Tuple[List[Position], List[List[int]], int, int]:
    metadata = maze_data.get("metadata", {})
    width = metadata.get("width")
    height = metadata.get("height")

    nodes = maze_data.get("nodes", [])
    edges = maze_data.get("edges", [])

    positions: Set[Position] = set()
    for node in nodes:
        if node.get("type") == "GHOSTSPACE":
            continue
        positions.add((node["x"], node["y"]))

    pos_list = sorted(positions)
    pos_to_idx = {pos: i for i, pos in enumerate(pos_list)}
    adjacency: List[List[int]] = [[] for _ in pos_list]

    for edge in edges:
        a = _parse_node_id(edge["from"])
        b = _parse_node_id(edge["to"])
        if a in pos_to_idx and b in pos_to_idx:
            ai = pos_to_idx[a]
            bi = pos_to_idx[b]
            adjacency[ai].append(bi)
            adjacency[bi].append(ai)

    if wrap_tunnels and width is not None and height is not None:
        for y in range(height):
            left = (0, y)
            right = (width - 1, y)
            if left in pos_to_idx and right in pos_to_idx:
                li = pos_to_idx[left]
                ri = pos_to_idx[right]
                adjacency[li].append(ri)
                adjacency[ri].append(li)

    return pos_list, adjacency, width or 0, height or 0


def _reachable_indices(adjacency: List[List[int]], start_idx: int) -> Set[int]:
    reachable: Set[int] = set()
    queue: Deque[int] = deque([start_idx])
    reachable.add(start_idx)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)
    return reachable


def _bfs_all_pairs(adjacency: List[List[int]]) -> List[List[int]]:
    count = len(adjacency)
    distances: List[List[int]] = [[-1] * count for _ in range(count)]
    for start in range(count):
        distances[start][start] = 0
        queue: Deque[int] = deque([start])
        while queue:
            current = queue.popleft()
            current_dist = distances[start][current]
            for neighbor in adjacency[current]:
                if distances[start][neighbor] == -1:
                    distances[start][neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances


def _shortest_path(adjacency: List[List[int]], start: int, goal: int) -> List[int]:
    if start == goal:
        return [start]
    queue: Deque[int] = deque([start])
    parent: Dict[int, int] = {start: -1}
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor in parent:
                continue
            parent[neighbor] = current
            if neighbor == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            queue.append(neighbor)
    raise ValueError("No path between nodes")


# ---------------------------------------------------------------------------
# Minimum Spanning Tree (Prim's algorithm)
# ---------------------------------------------------------------------------

def _prim_mst(
    distances: List[List[int]], nodes: List[int]
) -> Tuple[List[Tuple[int, int]], int]:
    if len(nodes) <= 1:
        return [], 0

    in_mst: Set[int] = set()
    key = {n: float("inf") for n in nodes}
    parent = {n: -1 for n in nodes}
    key[nodes[0]] = 0
    edges = []
    total_weight = 0

    for _ in range(len(nodes)):
        u = min((n for n in nodes if n not in in_mst), key=lambda n: key[n])
        in_mst.add(u)
        if parent[u] != -1:
            edges.append((parent[u], u))
            total_weight += distances[parent[u]][u]
        for v in nodes:
            if v not in in_mst and distances[u][v] != -1:
                if distances[u][v] < key[v]:
                    key[v] = distances[u][v]
                    parent[v] = u

    return edges, total_weight


# ---------------------------------------------------------------------------
# Held-Karp exact open walk (for very small mazes, <= 20 nodes)
# ---------------------------------------------------------------------------

def _held_karp_open(
    distances: List[List[int]], start_idx: int, nodes: List[int]
) -> Tuple[int, List[int]]:
    """
    Exact minimum-cost open walk (no return) via Held-Karp DP.
    O(2^n * n^2) — only for n <= 20.
    Returns (cost, ordered_node_sequence).
    """
    idx_map = {n: i for i, n in enumerate(nodes)}
    start = idx_map[start_idx]
    n = len(nodes)
    INF = float("inf")
    dp = [[INF] * n for _ in range(1 << n)]
    prev_node = [[-1] * n for _ in range(1 << n)]
    dp[1 << start][start] = 0
    for mask in range(1 << n):
        for last in range(n):
            if not (mask >> last & 1) or dp[mask][last] == INF:
                continue
            for nxt in range(n):
                if mask >> nxt & 1:
                    continue
                new_mask = mask | (1 << nxt)
                d = distances[nodes[last]][nodes[nxt]]
                if d < 0:
                    continue
                cost = dp[mask][last] + d
                if cost < dp[new_mask][nxt]:
                    dp[new_mask][nxt] = cost
                    prev_node[new_mask][nxt] = last
    full_mask = (1 << n) - 1
    best_cost = INF
    best_last = 0
    for last in range(n):
        if dp[full_mask][last] < best_cost:
            best_cost = dp[full_mask][last]
            best_last = last
    # Reconstruct path
    path = []
    mask = full_mask
    cur = best_last
    while cur != -1:
        path.append(nodes[cur])
        p = prev_node[mask][cur]
        mask ^= (1 << cur)
        cur = p
    path.reverse()
    return int(best_cost), path


# ---------------------------------------------------------------------------
# Local search helpers
# ---------------------------------------------------------------------------

def _route_length(distances: List[List[int]], order: List[int]) -> int:
    total = 0
    for i in range(len(order) - 1):
        d = distances[order[i]][order[i + 1]]
        if d < 0:
            raise ValueError("Route contains unreachable segments")
        total += d
    return total


def _greedy_order(
    distances: List[List[int]],
    start_idx: int,
    candidates: Set[int],
    seed: int = -1,
) -> List[int]:
    """
    Nearest-neighbour greedy ordering.
    seed >= 0: randomise among the 3 nearest at each step (diversification).
    """
    remaining: Set[int] = set(candidates)
    remaining.discard(start_idx)
    order = [start_idx]
    current = start_idx
    rng = _random.Random(seed) if seed >= 0 else None
    while remaining:
        if rng is not None and len(remaining) >= 3:
            top3 = sorted(remaining, key=lambda idx: distances[current][idx])[:3]
            next_idx = rng.choice(top3)
        else:
            next_idx = min(remaining, key=lambda idx: distances[current][idx])
        order.append(next_idx)
        remaining.remove(next_idx)
        current = next_idx
    return order


def _two_opt_scan(order: List[int], distances: List[List[int]]) -> List[int]:
    """
    Deterministic 2-opt: scan ALL (i, j) edge pairs, reverse the segment on
    improvement.  Repeats full passes until none improves.
    """
    n = len(order)
    if n <= 3:
        return order
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            di_prev = distances[order[i - 1]]
            di_cur = distances[order[i]]
            for j in range(i + 1, n):
                # delta = new_cost - old_cost
                # old: (order[i-1]→order[i]) + (order[j]→order[j+1] if exists)
                # new: (order[i-1]→order[j]) + (order[i]→order[j+1] if exists)
                delta = di_prev[order[j]] - di_prev[order[i]]
                if j + 1 < n:
                    delta += di_cur[order[j + 1]] - distances[order[j]][order[j + 1]]
                if delta < 0:
                    order[i : j + 1] = order[i : j + 1][::-1]
                    improved = True
    return order


def _or_opt_scan(order: List[int], distances: List[List[int]], chain_len: int) -> List[int]:
    """
    Or-opt: evaluate all relocations of every consecutive chain of *chain_len*
    nodes to every other gap in the route (also reversed).  Uses best-improvement
    strategy (pick the single best move, then restart scan).  The start node
    (index 0) is never relocated.
    """
    n = len(order)
    remaining_len = n - chain_len
    if remaining_len < 1 or chain_len >= n:
        return order

    improved = True
    while improved:
        improved = False
        best_delta = 0
        best_i = best_j = -1
        best_rev = False

        for i in range(1, n - chain_len + 1):  # i >= 1: protect start
            c_first = order[i]
            c_last = order[i + chain_len - 1]
            left_n = order[i - 1]
            right_n = order[i + chain_len] if i + chain_len < n else None

            r_save = distances[left_n][c_first]
            if right_n is not None:
                r_save += distances[c_last][right_n]
                bridge = distances[left_n][right_n]
            else:
                bridge = 0
            rem_delta = bridge - r_save  # net cost change from removing chain

            for j in range(remaining_len):
                # remaining[j] = order[j] if j < i else order[j + chain_len]
                rj = order[j] if j < i else order[j + chain_len]
                rj1_raw = j + 1
                if rj1_raw < remaining_len:
                    rj1: int | None = order[rj1_raw] if rj1_raw < i else order[rj1_raw + chain_len]
                else:
                    rj1 = None

                gap = distances[rj][rj1] if rj1 is not None else 0

                ins_fwd = (
                    distances[rj][c_first]
                    + (distances[c_last][rj1] if rj1 is not None else 0)
                    - gap
                )
                ins_rev = (
                    distances[rj][c_last]
                    + (distances[c_first][rj1] if rj1 is not None else 0)
                    - gap
                )

                for rev, ins_d in ((False, ins_fwd), (True, ins_rev)):
                    d = rem_delta + ins_d
                    if d < best_delta:
                        best_delta = d
                        best_i, best_j, best_rev = i, j, rev

        if best_delta < 0:
            improved = True
            chain = order[best_i : best_i + chain_len]
            if best_rev:
                chain = chain[::-1]
            remaining = order[:best_i] + order[best_i + chain_len :]
            order = remaining[: best_j + 1] + chain + remaining[best_j + 1 :]

    return order


def _three_opt_random(
    order: List[int], distances: List[List[int]], tries: int, seed: int = 0
) -> List[int]:
    """
    Random 3-opt: sample *tries* random triples of cut positions and evaluate
    all genuine 3-opt segment-swap reconnections (moves not reachable by 2-opt
    or Or-opt alone).  Keeps any improvement found.
    """
    n = len(order)
    if n < 6 or tries <= 0:
        return order
    rng = _random.Random(seed)
    best = order[:]
    best_len = _route_length(distances, best)

    for _ in range(tries):
        i, j, k = sorted(rng.sample(range(1, n), 3))
        A, B, C, D = best[:i], best[i:j], best[j:k], best[k:]
        # Genuine 3-opt reconnections (segment permutations/reversals)
        for cand in (
            A + C + B + D,             # swap B and C
            A + C + B[::-1] + D,       # swap, reverse B
            A + C[::-1] + B + D,       # swap, reverse C
            A + B[::-1] + C[::-1] + D, # reverse both (double 2-opt)
        ):
            cand_len = _route_length(distances, cand)
            if cand_len < best_len:
                best = cand
                best_len = cand_len

    return best


def _full_local_search(
    order: List[int], distances: List[List[int]], three_opt_tries: int
) -> Tuple[int, List[int]]:
    """
    Full local-search pipeline applied to a single starting order:
      1. Deterministic 2-opt scan  (to convergence)
      2. Or-opt chain lengths 1, 2, 3  (best-improvement, to convergence each)
      3. Another 2-opt scan  (Or-opt can open new 2-opt gains)
      4. Random 3-opt  (escape remaining local optima)
      5. Final 2-opt pass after any 3-opt moves
    """
    order = _two_opt_scan(order, distances)
    for k in (1, 2, 3):
        order = _or_opt_scan(order, distances, k)
    order = _two_opt_scan(order, distances)
    if three_opt_tries > 0:
        order = _three_opt_random(order, distances, three_opt_tries)
        order = _two_opt_scan(order, distances)
    return _route_length(distances, order), order


def _best_path(
    distances: List[List[int]],
    start_idx: int,
    reachable: Set[int],
    num_restarts: int,
) -> Tuple[int, List[int]]:
    """
    Run *num_restarts* greedy starts (first deterministic, rest randomised)
    each through the full local-search pipeline; return the best (steps, order).
    """
    n = len(reachable)
    three_opt_tries = n * 20 if n >= 6 else 0
    best_steps: int = -1
    best_order: List[int] = []
    for attempt in range(num_restarts):
        seed = -1 if attempt == 0 else attempt
        order = _greedy_order(distances, start_idx, reachable, seed=seed)
        steps, order = _full_local_search(order, distances, three_opt_tries)
        if best_steps < 0 or steps < best_steps:
            best_steps = steps
            best_order = order
    return best_steps, best_order


# ---------------------------------------------------------------------------
# New signal helpers
# ---------------------------------------------------------------------------


def count_turns(path_positions: List[Position]) -> int:
    """Count direction changes in a tile-position sequence (O(P))."""
    turns = 0
    for i in range(1, len(path_positions) - 1):
        dx0 = path_positions[i][0] - path_positions[i - 1][0]
        dy0 = path_positions[i][1] - path_positions[i - 1][1]
        dx1 = path_positions[i + 1][0] - path_positions[i][0]
        dy1 = path_positions[i + 1][1] - path_positions[i][1]
        if (dx0, dy0) != (dx1, dy1):
            turns += 1
    return turns


def find_trap_corridors(
    pos_list: List[Position],
    adjacency: List[List[int]],
    min_interior: int = 1,
) -> Tuple[List[List[Position]], int, int]:
    """Find all valid trap corridors in the maze graph (O(N)).

    A trap corridor is a maximal chain of degree-2 interior nodes bounded at
    both ends by a distinct junction node of degree >= 3.

    Args:
        min_interior: only corridors whose interior node count is strictly
                      greater than this value are included in the results.

    Returns:
        corridors  -- list of corridors, each as [(x,y), ...] from v0 to vn
        longest    -- interior node count of the longest corridor (0 if none)
        num_traps  -- count of valid corridors
    """
    n = len(pos_list)
    degree = [len(adjacency[i]) for i in range(n)]
    visited = [False] * n
    corridors: List[List[Position]] = []
    longest = 0

    for seed in range(n):
        if degree[seed] != 2 or visited[seed]:
            continue

        # Mark seed visited so loop-detection works
        visited[seed] = True

        # Walk in the direction of neighbor 0 (left arm)
        chain_left: List[int] = []
        prev, cur = seed, adjacency[seed][0]
        while degree[cur] == 2 and not visited[cur]:
            visited[cur] = True
            chain_left.append(cur)
            nxt = adjacency[cur][0] if adjacency[cur][1] == prev else adjacency[cur][1]
            prev, cur = cur, nxt
        v_left = cur  # boundary node (degree >= 3, or visited if loop)

        # Walk in the direction of neighbor 1 (right arm)
        chain_right: List[int] = []
        prev, cur = seed, adjacency[seed][1]
        while degree[cur] == 2 and not visited[cur]:
            visited[cur] = True
            chain_right.append(cur)
            nxt = adjacency[cur][0] if adjacency[cur][1] == prev else adjacency[cur][1]
            prev, cur = cur, nxt
        v_right = cur  # boundary node

        # Interior = chain_left reversed + seed + chain_right
        interior = list(reversed(chain_left)) + [seed] + chain_right

        # Validity filters
        if degree[v_left] < 3 or degree[v_right] < 3:
            continue
        if v_left == v_right:
            continue  # loop — same junction on both sides
        if len(interior) <= min_interior:
            continue  # too short — skip

        # Record corridor as position list from v0 to vn inclusive
        corridor_indices = [v_left] + interior + [v_right]
        corridors.append([pos_list[i] for i in corridor_indices])
        interior_len = len(interior)
        if interior_len > longest:
            longest = interior_len

    return corridors, longest, len(corridors)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _expand_node_order(
    adjacency: List[List[int]],
    pos_list: List[Position],
    order: List[int],
) -> List[Position]:
    """Expand an ordered list of node indices to a full tile-position sequence."""
    path_positions: List[Position] = []
    for i in range(len(order) - 1):
        segment = _shortest_path(adjacency, order[i], order[i + 1])
        if i > 0:
            segment = segment[1:]
        path_positions.extend(pos_list[n] for n in segment)
    return path_positions


def compute_optimal_path(
    maze_data: Dict,
    start: Position,
    wrap_tunnels: bool,
    force_held_karp: bool = False,
) -> Tuple[int, Dict]:
    """
    Compute the best-known open-walk visiting all nodes from start.

    For <= 20 reachable nodes: Held-Karp exact DP (provably optimal).
    For larger mazes: greedy nearest-neighbour + 2-opt (best practical approx).

    Returns (steps, debug_info).
    """
    pos_list, adjacency, width, height = _build_graph(maze_data, wrap_tunnels)
    pos_to_idx = {pos: i for i, pos in enumerate(pos_list)}

    if start not in pos_to_idx:
        raise ValueError("Start position %s is not walkable in this maze" % (start,))

    start_idx = pos_to_idx[start]
    reachable = _reachable_indices(adjacency, start_idx)
    food_count = len(reachable)
    distances = _bfs_all_pairs(adjacency)

    # MST weight = valid lower bound on the optimal open walk length
    nodes = sorted(reachable)
    _, mst_weight = _prim_mst(distances, nodes)

    if force_held_karp or food_count <= 20:
        if food_count > 20:
            raise ValueError(
                "Held-Karp requires <= 20 nodes, got %d. Remove --held-karp flag." % food_count
            )
        steps, _ = _held_karp_open(distances, start_idx, nodes)
        method = "held-karp-exact"
    else:
        if food_count <= 50:
            num_restarts = 10
        elif food_count <= 100:
            num_restarts = 8
        elif food_count <= 200:
            num_restarts = 5
        else:
            num_restarts = 3
        steps, _ = _best_path(distances, start_idx, reachable, num_restarts)
        method = "ms-or3opt"

    return steps, {
        "width": width,
        "height": height,
        "food_count": food_count,
        "method": method,
        "mst_lower_bound": mst_weight,
    }


def compute_path_for_visualization(
    maze_data: Dict,
    start: Position,
    wrap_tunnels: bool,
) -> Tuple[int, List[Position], Dict]:
    """
    Compute the best open-walk order and expand to tile positions for animation.
    Also reports the MST lower bound in debug output.
    """
    pos_list, adjacency, width, height = _build_graph(maze_data, wrap_tunnels)
    pos_to_idx = {pos: i for i, pos in enumerate(pos_list)}

    if start not in pos_to_idx:
        raise ValueError("Start position %s is not walkable in this maze" % (start,))

    start_idx = pos_to_idx[start]
    reachable = _reachable_indices(adjacency, start_idx)
    food_count = len(reachable)
    distances = _bfs_all_pairs(adjacency)
    nodes = sorted(reachable)
    _, mst_weight = _prim_mst(distances, nodes)

    if food_count <= 20:
        steps, order = _held_karp_open(distances, start_idx, nodes)
        method = "held-karp-exact"
    else:
        if food_count <= 50:
            num_restarts = 10
        elif food_count <= 100:
            num_restarts = 8
        elif food_count <= 200:
            num_restarts = 5
        else:
            num_restarts = 3
        steps, order = _best_path(distances, start_idx, reachable, num_restarts)
        method = "ms-or3opt"

    path_positions = _expand_node_order(adjacency, pos_list, order)
    if not path_positions:
        path_positions = [start]

    return steps, path_positions, {
        "width": width,
        "height": height,
        "food_count": food_count,
        "steps": steps,
        "method": method,
        "mst_lower_bound": mst_weight,
    }


def compute_complexity_score(
    maze_data: Dict,
    start: Position,
    wrap_tunnels: bool,
    force_held_karp: bool = False,
) -> Dict:
    # 5.1 Build graph once
    pos_list, adjacency, width, height = _build_graph(maze_data, wrap_tunnels)
    pos_to_idx = {pos: i for i, pos in enumerate(pos_list)}

    if start not in pos_to_idx:
        raise ValueError("Start position %s is not walkable in this maze" % (start,))

    start_idx = pos_to_idx[start]
    reachable = _reachable_indices(adjacency, start_idx)
    food_count = len(reachable)
    nodes = sorted(reachable)

    # 5.2 BFS distances once
    distances = _bfs_all_pairs(adjacency)

    # 5.3 Compute optimal path
    if force_held_karp or food_count <= 20:
        if food_count > 20:
            raise ValueError(
                "Held-Karp requires <= 20 nodes, got %d. Remove --held-karp flag." % food_count
            )
        steps, order = _held_karp_open(distances, start_idx, nodes)
        method = "held-karp-exact"
    else:
        if food_count <= 50:
            num_restarts = 10
        elif food_count <= 100:
            num_restarts = 8
        elif food_count <= 200:
            num_restarts = 5
        else:
            num_restarts = 3
        steps, order = _best_path(distances, start_idx, reachable, num_restarts)
        method = "ms-or3opt"

    # 5.4 Count tunnel rows (horizontal wrap-around passages)
    num_tunnels: int = sum(
        1 for y in range(height)
        if (0, y) in pos_to_idx and (width - 1, y) in pos_to_idx
    )

    # 5.5 Find trap corridors
    trap_corridors, longest_trap, num_traps = find_trap_corridors(
        pos_list, adjacency, min_interior=TRAP_MIN_INTERIOR
    )
    avg_trap_len: float = (
        sum(len(c) for c in trap_corridors) / num_traps if num_traps > 0 else 0.0
    )

    # 5.7 Compute score
    # Each signal: z-normalise then apply tanh  →  (-1, +1)
    # Negative = easier than average, positive = harder.
    # Score mapped from mean(3 signals) ∈ (-1,+1) to [0, 10] via (x+1)/2*10.
    steps_ratio = steps / food_count if food_count > 0 else MEAN_STEPS_RATIO
    s_steps    = math.tanh((steps_ratio  - MEAN_STEPS_RATIO) / STD_STEPS_RATIO)
    s_trap_len = math.tanh((longest_trap - MEAN_TRAP_LEN)    / STD_TRAP_LEN)
    s_trap_cnt = math.tanh((num_traps    - MEAN_TRAP_CNT)    / STD_TRAP_CNT)

    combined = W_STEPS * s_steps + W_TRAP_LEN * s_trap_len + W_TRAP_CNT * s_trap_cnt
    score = (combined + 1.0) / 2.0 * 10.0

    # 5.9 Return updated dict
    return {
        "score":        round(score, 4),
        "steps":        steps,
        "longest_trap": longest_trap,
        "avg_trap_len": round(avg_trap_len, 4),
        "num_traps":    num_traps,
        "num_tunnels":  num_tunnels,
    }


def add_complexity_to_maze_json(
    maze_json_path: str,
    start: Position,
    wrap_tunnels: bool,
    force_held_karp: bool = False,
) -> Dict:
    with open(maze_json_path, "r") as f:
        maze_data = json.load(f)

    score_data = compute_complexity_score(maze_data, start, wrap_tunnels, force_held_karp)
    metadata = maze_data.setdefault("metadata", {})
    metadata["complexity_score"] = score_data

    with open(maze_json_path, "w") as f:
        json.dump(maze_data, f, indent=2)

    return maze_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute maze complexity score (open walk) and store it in maze.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Default: multi-start greedy + Or-opt(1,2,3) + random 3-opt (best practical approximation).
For mazes with <= 20 reachable nodes: Held-Karp gives the provably exact optimum.
        """,
    )
    parser.add_argument("maze_json", help="Path to maze.json")
    parser.add_argument(
        "--no-tunnel-wrap",
        action="store_true",
        help="Disable left-right tunnel wrapping",
    )
    parser.add_argument(
        "--held-karp",
        action="store_true",
        help="Use Held-Karp exact DP (only for mazes with <= 20 nodes)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    wrap_tunnels = not args.no_tunnel_wrap
    updated = add_complexity_to_maze_json(
        args.maze_json,
        start=(13, 23),
        wrap_tunnels=wrap_tunnels,
        force_held_karp=args.held_karp,
    )

    if args.verbose:
        score = updated.get("metadata", {}).get("complexity_score", {})
        print(
            "Complexity score: %s (steps: %s, method: %s)"
            % (
                score.get("score"),
                score.get("steps"),
                score.get("method"),
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
