#!/usr/bin/env python3
"""
Statistical analysis of maze difficulty signals.

Generates N mazes in-memory, computes complexity signals for each, and
reports the full distribution of longest_trap, num_traps, and num_turns
so that MAX_TRAP_LEN, MAX_NUM_TRAPS, MAX_TURNS can be calibrated.

Usage:
    python3 analyse_signals.py [--n N] [--seed SEED] [--start-seed S]
"""

import argparse
import random
import sys

from src.pacman_map import PacmanMap
from src.maze_complexity import (
    _build_graph,
    _reachable_indices,
    _bfs_all_pairs,
    _greedy_order,
    _route_length,
    _held_karp_open,
    _expand_node_order,
    count_turns,
    find_trap_corridors,
)

START: tuple = (13, 23)
WRAP_TUNNELS: bool = True


def _analyse_maze(seed: int) -> dict | None:
    """Generate one maze and return its signal values, or None on failure."""
    try:
        maze = PacmanMap(seed=seed)
        maze.generate()
        maze_data = maze.export_to_graph_json(None)
    except Exception as e:
        print(f"  [seed {seed}] generation failed: {e}", file=sys.stderr)
        return None

    try:
        pos_list, adjacency, _, _ = _build_graph(maze_data, WRAP_TUNNELS)
        pos_to_idx = {pos: i for i, pos in enumerate(pos_list)}
        if START not in pos_to_idx:
            return None

        start_idx = pos_to_idx[START]
        reachable = _reachable_indices(adjacency, start_idx)
        food_count = len(reachable)
        nodes = sorted(reachable)
        distances = _bfs_all_pairs(adjacency)

        if food_count <= 20:
            steps, order = _held_karp_open(distances, start_idx, nodes)
        else:
            # Fast single greedy pass — good enough for distribution calibration
            order = _greedy_order(distances, start_idx, reachable)
            steps = _route_length(distances, order)

        path_positions = _expand_node_order(adjacency, pos_list, order)
        if not path_positions:
            path_positions = [START]

        num_turns = count_turns(path_positions)
        _, longest_trap, num_traps = find_trap_corridors(pos_list, adjacency)

        return {
            "seed":         seed,
            "food_count":   food_count,
            "steps":        steps,
            "longest_trap": longest_trap,
            "num_traps":    num_traps,
            "num_turns":    num_turns,
        }
    except Exception as e:
        print(f"  [seed {seed}] scoring failed: {e}", file=sys.stderr)
        return None


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    data = sorted(data)
    idx = (len(data) - 1) * p / 100.0
    lo = int(idx)
    hi = lo + 1
    frac = idx - lo
    if hi >= len(data):
        return data[lo]
    return data[lo] * (1 - frac) + data[hi] * frac


def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = var ** 0.5
    return {
        "n":    n,
        "min":  min(values),
        "p25":  _percentile(values, 25),
        "p50":  _percentile(values, 50),
        "mean": mean,
        "p75":  _percentile(values, 75),
        "p90":  _percentile(values, 90),
        "p95":  _percentile(values, 95),
        "p99":  _percentile(values, 99),
        "max":  max(values),
        "std":  std,
    }


def _print_stats(label: str, values: list[float], current_cap: int | None = None) -> None:
    s = _stats(values)
    print(f"\n  {label}")
    print(f"    n={s['n']}  min={s['min']:.1f}  max={s['max']:.1f}  "
          f"mean={s['mean']:.1f}  std={s['std']:.1f}")
    print(f"    p25={s['p25']:.1f}  p50={s['p50']:.1f}  p75={s['p75']:.1f}  "
          f"p90={s['p90']:.1f}  p95={s['p95']:.1f}  p99={s['p99']:.1f}")
    if current_cap is not None:
        pct_below = 100 * sum(1 for v in values if v <= current_cap) / len(values)
        print(f"    current cap={current_cap}  →  {pct_below:.1f}% of mazes are at or below it")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse maze difficulty signal distributions")
    parser.add_argument("-n", "--n", type=int, default=200,
                        help="Number of mazes to generate (default: 200)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for selecting maze seeds (default: 0)")
    parser.add_argument("--start-seed", type=int, default=None,
                        help="Use consecutive seeds starting from this value instead of random")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    if args.start_seed is not None:
        maze_seeds = list(range(args.start_seed, args.start_seed + args.n))
    else:
        maze_seeds = [rng.randint(0, 10_000_000) for _ in range(args.n)]

    print(f"Analysing {args.n} mazes …")
    results = []
    for i, seed in enumerate(maze_seeds):
        r = _analyse_maze(seed)
        if r:
            results.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{args.n} done ({len(results)} successful)", end="\r")

    print(f"\n{len(results)}/{args.n} mazes scored successfully.")

    longest_traps = [r["longest_trap"] for r in results]
    num_traps_lst = [r["num_traps"]    for r in results]
    num_turns_lst = [r["num_turns"]    for r in results]

    print("\n" + "=" * 60)
    print("  SIGNAL DISTRIBUTIONS")
    print("=" * 60)

    from src.maze_complexity import MAX_TRAP_LEN, MAX_NUM_TRAPS, MAX_TURNS
    _print_stats("longest_trap", longest_traps, MAX_TRAP_LEN)
    _print_stats("num_traps",    num_traps_lst,  MAX_NUM_TRAPS)
    _print_stats("num_turns",    num_turns_lst,  MAX_TURNS)

    print("\n" + "=" * 60)
    print("  SUGGESTED CAPS (rounded up to nearest integer)")
    print("=" * 60)
    for label, values, current in (
        ("MAX_TRAP_LEN  (p95)", longest_traps, MAX_TRAP_LEN),
        ("MAX_NUM_TRAPS (p95)", num_traps_lst,  MAX_NUM_TRAPS),
        ("MAX_TURNS     (p95)", num_turns_lst,  MAX_TURNS),
    ):
        p95 = _percentile(values, 95)
        suggested = max(1, int(p95) + (1 if p95 % 1 else 0))
        marker = "  ← UNCHANGED" if suggested == current else f"  ← was {current}"
        print(f"    {label}:  p95={p95:.1f}  →  {suggested}{marker}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
