#!/usr/bin/env python3
"""
Compute a maze complexity score based on a FoodSearch-style traversal.
The score is the number of steps to visit all walkable tiles, assuming food
on every path tile and a fixed start position.
"""

import argparse
import json
import os
import random
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
        if distances[current][next_idx] < 0:
            raise ValueError("Some food tiles are unreachable from the start position")
        order.append(next_idx)
        remaining.remove(next_idx)
        current = next_idx

    return order


def _determine_swap_iterations(food_count: int) -> int:
    """
    Automatically determine the number of improvement iterations based on maze size.
    Uses a formula that scales with the number of food tiles for near-optimal results.
    """
    if food_count <= 10:
        return food_count * 50
    elif food_count <= 50:
        return food_count * 30
    elif food_count <= 100:
        return food_count * 20
    else:
        return food_count * 10


def _two_opt_improve(order: List[int], distances: List[List[int]], iterations: int) -> Tuple[List[int], int]:
    """
    Improve route using 2-opt local search algorithm.
    2-opt is more effective than random swaps for finding near-optimal solutions.
    """
    if iterations <= 0 or len(order) <= 3:
        return order, 0

    rng = random.Random(0)
    improvements = 0
    best_order = order[:]
    best_length = _route_length(distances, best_order)

    for _ in range(iterations):
        # Select two random edges in the tour
        i = rng.randint(1, len(best_order) - 2)
        j = rng.randint(i + 1, len(best_order) - 1)
        
        # Reverse the segment between i and j (2-opt swap)
        candidate = best_order[:i] + best_order[i:j+1][::-1] + best_order[j+1:]
        candidate_length = _route_length(distances, candidate)

        if candidate_length < best_length:
            best_order = candidate
            best_length = candidate_length
            improvements += 1

    return best_order, improvements


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
    raise ValueError("No path between nodes in route")


def compute_greedy_path(maze_data: Dict, start: Position, wrap_tunnels: bool, swap_iterations: int = -1) -> Tuple[int, List[Position], Dict]:
    pos_list, adjacency, width, height = _build_graph(maze_data, wrap_tunnels)
    pos_to_idx = {pos: i for i, pos in enumerate(pos_list)}

    if start not in pos_to_idx:
        raise ValueError("Start position %s is not walkable in this maze" % (start,))

    start_idx = pos_to_idx[start]
    food_count = len(pos_list)
    
    # Automatically determine swap iterations if not specified
    if swap_iterations < 0:
        swap_iterations = _determine_swap_iterations(food_count)
    
    reachable = _reachable_indices(adjacency, start_idx)
    distances = _bfs_all_pairs(adjacency)
    order = _greedy_order(distances, start_idx, reachable)
    order, improvements = _two_opt_improve(order, distances, swap_iterations)

    steps = _route_length(distances, order)

    path_positions: List[Position] = []
    for idx in range(len(order) - 1):
        segment = _shortest_path(adjacency, order[idx], order[idx + 1])
        if idx > 0:
            segment = segment[1:]
        path_positions.extend(pos_list[node] for node in segment)

    if not path_positions:
        path_positions = [start]

    return steps, path_positions, {
        "width": width,
        "height": height,
        "food_count": food_count,
        "swap_iterations": swap_iterations,
        "improvements_applied": improvements,
    }


def compute_complexity_score(maze_data: Dict, start: Position, wrap_tunnels: bool) -> Dict:
    steps, _, debug = compute_greedy_path(maze_data, start, wrap_tunnels)
    food_count = debug["food_count"]
    
    # Calculate score as (food_count / steps) * 10, capped between 0-10
    score = (food_count / steps * 10) if steps > 0 else 10.0
    score = min(10.0, score)  # Cap at 10
    
    score_data = {
        "start": {"x": start[0], "y": start[1]},
        "food_count": food_count,
        "steps": steps,
        "score": round(score, 2),
        "method": "greedy-nearest-with-2opt",
        "wrap_tunnels": wrap_tunnels,
        "width": debug["width"],
        "height": debug["height"],
        "swap_iterations": debug["swap_iterations"],
        "improvements_applied": debug["improvements_applied"],
    }
    return score_data


def add_complexity_to_maze_json(maze_json_path: str, start: Position, wrap_tunnels: bool) -> Dict:
    with open(maze_json_path, "r") as f:
        maze_data = json.load(f)

    score_data = compute_complexity_score(maze_data, start, wrap_tunnels)
    metadata = maze_data.setdefault("metadata", {})
    metadata["complexity_score"] = score_data

    with open(maze_json_path, "w") as f:
        json.dump(maze_data, f, indent=2)

    return maze_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute maze complexity score and store it in maze.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("maze_json", help="Path to maze.json")
    parser.add_argument(
        "--no-tunnel-wrap",
        action="store_true",
        help="Disable left-right tunnel wrapping",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    wrap_tunnels = not args.no_tunnel_wrap
    updated = add_complexity_to_maze_json(
        args.maze_json,
        start=(13, 23),
        wrap_tunnels=wrap_tunnels,
    )

    if args.verbose:
        score = updated.get("metadata", {}).get("complexity_score", {})
        print("Complexity score: %s (steps: %s, food: %s)" % (
            score.get("score"),
            score.get("steps"),
            score.get("food_count")
        ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
