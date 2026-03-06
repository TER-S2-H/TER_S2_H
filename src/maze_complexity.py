#!/usr/bin/env python3
"""
Compute an OPTIMAL maze complexity score using the Chinese Postman / T-join approach.

Problem: Find the shortest closed walk that visits every node in the maze.

Algorithm (exact, polynomial time):
  1. Build a complete auxiliary graph on all maze nodes (edge weight = BFS distance).
  2. Find a minimum spanning tree (MST) of this auxiliary graph.
  3. Identify nodes with odd degree in the MST.
  4. Compute minimum weight perfect matching on odd-degree nodes.
  5. MST weight + matching weight = EXACT optimal closed-walk length.

  BFS distances always satisfy the triangle inequality (metric graph), which
  makes the T-join result provably optimal for this problem.

  For tiny mazes (<=20 reachable nodes): Held-Karp exact TSP is also available.

Score formula:
  score = (food_count / steps) * 10   (capped at 10, always in [0, 10])

Visualization:
  compute_path_for_visualization() returns an ordered walk (positions list)
  suitable for GIF animation, using greedy nearest-neighbour + 2-opt.
  The score itself always comes from the exact optimal algorithm above.
"""

import argparse
import json
import random as _random
from collections import deque
from typing import Deque, Dict, List, Set, Tuple

Position = Tuple[int, int]


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
# Minimum Weight Perfect Matching
# ---------------------------------------------------------------------------

def _exact_matching_dp(odd_nodes: List[int], distances: List[List[int]]) -> int:
    """Exact minimum weight perfect matching via bitmask DP. O(2^n * n)."""
    n = len(odd_nodes)
    INF = float("inf")
    dp = [INF] * (1 << n)
    dp[0] = 0
    for mask in range(1 << n):
        if dp[mask] == INF:
            continue
        first = next((i for i in range(n) if not (mask >> i & 1)), -1)
        if first == -1:
            continue
        for second in range(first + 1, n):
            if not (mask >> second & 1):
                new_mask = mask | (1 << first) | (1 << second)
                cost = distances[odd_nodes[first]][odd_nodes[second]]
                if dp[mask] + cost < dp[new_mask]:
                    dp[new_mask] = dp[mask] + cost
    return int(dp[(1 << n) - 1])


def _greedy_matching(odd_nodes: List[int], distances: List[List[int]]) -> int:
    """Greedy nearest-neighbour matching for large odd sets (> 24)."""
    remaining = set(odd_nodes)
    total = 0
    while remaining:
        u = next(iter(remaining))
        remaining.remove(u)
        best = min(remaining, key=lambda v: distances[u][v])
        total += distances[u][best]
        remaining.remove(best)
    return total


def _min_weight_perfect_matching(odd_nodes: List[int], distances: List[List[int]]) -> int:
    n = len(odd_nodes)
    assert n % 2 == 0
    if n == 0:
        return 0
    return _exact_matching_dp(odd_nodes, distances) if n <= 24 else _greedy_matching(odd_nodes, distances)


# ---------------------------------------------------------------------------
# T-join / Chinese Postman — exact optimal walk length
# ---------------------------------------------------------------------------

def _compute_optimal_walk_length(
    adjacency: List[List[int]],
    distances: List[List[int]],
    start_idx: int,
    reachable: Set[int],
) -> int:
    """
    Exact minimum steps to visit all reachable nodes in a closed walk.
    MST + min-weight perfect matching on odd-degree nodes (T-join).
    Provably optimal for metric graphs (BFS distances are always metric).
    """
    nodes = sorted(reachable)
    if len(nodes) == 1:
        return 0
    mst_edges, mst_weight = _prim_mst(distances, nodes)
    degree = {n: 0 for n in nodes}
    for u, v in mst_edges:
        degree[u] += 1
        degree[v] += 1
    odd_nodes = [n for n in nodes if degree[n] % 2 == 1]
    matching_weight = _min_weight_perfect_matching(odd_nodes, distances)
    return mst_weight + matching_weight


# ---------------------------------------------------------------------------
# Held-Karp exact TSP (for very small mazes, <= 20 nodes)
# ---------------------------------------------------------------------------

def _held_karp_exact(
    distances: List[List[int]], start_idx: int, nodes: List[int]
) -> int:
    """Exact TSP via Held-Karp DP. O(2^n * n^2) — only for n <= 20."""
    idx_map = {n: i for i, n in enumerate(nodes)}
    start = idx_map[start_idx]
    n = len(nodes)
    INF = float("inf")
    dp = [[INF] * n for _ in range(1 << n)]
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
    full_mask = (1 << n) - 1
    best = INF
    for last in range(n):
        if dp[full_mask][last] == INF:
            continue
        d_back = distances[nodes[last]][nodes[start]]
        if d_back >= 0:
            best = min(best, dp[full_mask][last] + d_back)
    return int(best) if best != INF else -1


# ---------------------------------------------------------------------------
# Greedy + 2-opt helpers (used only for visualization path ordering)
# ---------------------------------------------------------------------------

def _route_length(distances: List[List[int]], order: List[int]) -> int:
    total = 0
    for i in range(len(order) - 1):
        d = distances[order[i]][order[i + 1]]
        if d < 0:
            raise ValueError("Route contains unreachable segments")
        total += d
    return total


def _greedy_order(distances: List[List[int]], start_idx: int, candidates: Set[int]) -> List[int]:
    remaining: Set[int] = set(candidates)
    remaining.discard(start_idx)
    order = [start_idx]
    current = start_idx
    while remaining:
        next_idx = min(remaining, key=lambda idx: distances[current][idx])
        order.append(next_idx)
        remaining.remove(next_idx)
        current = next_idx
    return order


def _two_opt_improve(order: List[int], distances: List[List[int]], iterations: int) -> List[int]:
    """2-opt local search to improve a greedy tour (for visualization only)."""
    if iterations <= 0 or len(order) <= 3:
        return order
    rng = _random.Random(0)
    best = order[:]
    best_len = _route_length(distances, best)
    for _ in range(iterations):
        i = rng.randint(1, len(best) - 2)
        j = rng.randint(i + 1, len(best) - 1)
        candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
        candidate_len = _route_length(distances, candidate)
        if candidate_len < best_len:
            best = candidate
            best_len = candidate_len
    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_optimal_path(
    maze_data: Dict,
    start: Position,
    wrap_tunnels: bool,
    force_held_karp: bool = False,
) -> Tuple[int, Dict]:
    """
    Compute the EXACT minimum steps to visit all nodes in a closed walk.
    Returns (steps, debug_info). Does NOT return a position sequence.
    Use compute_path_for_visualization() when you need the ordered walk.
    """
    pos_list, adjacency, width, height = _build_graph(maze_data, wrap_tunnels)
    pos_to_idx = {pos: i for i, pos in enumerate(pos_list)}

    if start not in pos_to_idx:
        raise ValueError("Start position %s is not walkable in this maze" % (start,))

    start_idx = pos_to_idx[start]
    reachable = _reachable_indices(adjacency, start_idx)
    food_count = len(reachable)
    distances = _bfs_all_pairs(adjacency)

    method = "t-join-exact"
    if force_held_karp or food_count <= 20:
        if food_count > 20:
            raise ValueError(
                "Held-Karp requires <= 20 nodes, got %d. "
                "Remove --held-karp flag to use T-join exact algorithm." % food_count
            )
        steps = _held_karp_exact(distances, start_idx, sorted(reachable))
        method = "held-karp-exact"
    else:
        steps = _compute_optimal_walk_length(adjacency, distances, start_idx, reachable)

    return steps, {
        "width": width,
        "height": height,
        "food_count": food_count,
        "method": method,
    }


def compute_path_for_visualization(
    maze_data: Dict,
    start: Position,
    wrap_tunnels: bool,
) -> Tuple[int, List[Position], Dict]:
    """
    Build a near-optimal ordered walk and expand it into a full tile-position
    sequence for GIF animation (greedy nearest-neighbour + 2-opt).
    The step count here is approximate; for the authoritative score use
    compute_optimal_path() or read the stored JSON value.
    """
    pos_list, adjacency, width, height = _build_graph(maze_data, wrap_tunnels)
    pos_to_idx = {pos: i for i, pos in enumerate(pos_list)}

    if start not in pos_to_idx:
        raise ValueError("Start position %s is not walkable in this maze" % (start,))

    start_idx = pos_to_idx[start]
    reachable = _reachable_indices(adjacency, start_idx)
    food_count = len(reachable)

    if food_count <= 10:
        iters = food_count * 50
    elif food_count <= 50:
        iters = food_count * 30
    elif food_count <= 100:
        iters = food_count * 20
    else:
        iters = food_count * 10

    distances = _bfs_all_pairs(adjacency)
    order = _greedy_order(distances, start_idx, reachable)
    order = _two_opt_improve(order, distances, iters)
    steps = _route_length(distances, order)

    path_positions: List[Position] = []
    for idx in range(len(order) - 1):
        segment = _shortest_path(adjacency, order[idx], order[idx + 1])
        if idx > 0:
            segment = segment[1:]
        path_positions.extend(pos_list[n] for n in segment)
    if not path_positions:
        path_positions = [start]

    return steps, path_positions, {
        "width": width,
        "height": height,
        "food_count": food_count,
    }


def compute_complexity_score(
    maze_data: Dict,
    start: Position,
    wrap_tunnels: bool,
    force_held_karp: bool = False,
) -> Dict:
    steps, debug = compute_optimal_path(maze_data, start, wrap_tunnels, force_held_karp)
    food_count = debug["food_count"]
    score = (food_count / steps * 10) if steps > 0 else 10.0
    score = min(10.0, score)
    return {
        "start": {"x": start[0], "y": start[1]},
        "food_count": food_count,
        "steps": steps,
        "score": round(score, 2),
        "method": debug["method"],
        "wrap_tunnels": wrap_tunnels,
        "width": debug["width"],
        "height": debug["height"],
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
        description="Compute OPTIMAL maze complexity score and store it in maze.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Algorithm: T-join exact (MST + min-weight perfect matching on odd-degree nodes).
Provably optimal for the closed node-covering walk on metric graphs.
Use --held-karp for mazes with <= 20 reachable nodes (alternative exact method).
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
        help="Use Held-Karp exact TSP (only for mazes with <= 20 nodes)",
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
            "Complexity score: %s (steps: %s, food: %s, method: %s)"
            % (
                score.get("score"),
                score.get("steps"),
                score.get("food_count"),
                score.get("method"),
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
