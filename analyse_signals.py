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
    MIN_TRAVERSAL, MAX_TRAVERSAL,
    MIN_TRAP_LEN,  MAX_TRAP_LEN,
    MIN_NUM_TRAPS, MAX_NUM_TRAPS,
    MIN_TURNS,     MAX_TURNS,
    W_TRAP_LEN, W_TRAP_CNT, W_TURNS,
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

        # Normalized signals (same formula as compute_complexity_score)
        raw_ratio   = food_count / steps if steps > 0 else MIN_TRAVERSAL
        s_traversal = (raw_ratio   - MIN_TRAVERSAL) / (MAX_TRAVERSAL - MIN_TRAVERSAL)
        s_traversal = 1.0 - max(0.0, min(s_traversal, 1.0))
        s_trap_len  = (longest_trap - MIN_TRAP_LEN)  / (MAX_TRAP_LEN  - MIN_TRAP_LEN)
        s_trap_len  = max(0.0, min(s_trap_len,  1.0))
        s_trap_cnt  = (num_traps    - MIN_NUM_TRAPS) / (MAX_NUM_TRAPS - MIN_NUM_TRAPS)
        s_trap_cnt  = max(0.0, min(s_trap_cnt,  1.0))
        s_turns     = (num_turns    - MIN_TURNS)     / (MAX_TURNS     - MIN_TURNS)
        s_turns     = max(0.0, min(s_turns,     1.0))
        raw   = (W_TRAP_LEN * s_trap_len + W_TRAP_CNT * s_trap_cnt + W_TURNS * s_turns)
        score = min(raw * 10, 10.0)

        return {
            "seed":         seed,
            "food_count":   food_count,
            "steps":        steps,
            "longest_trap": longest_trap,
            "num_traps":    num_traps,
            "num_turns":    num_turns,
            "s_traversal":  s_traversal,
            "s_trap_len":   s_trap_len,
            "s_trap_cnt":   s_trap_cnt,
            "s_turns":      s_turns,
            "score":        score,
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


def _print_stats(label: str, values: list[float], current_cap: int | None = None, fmt: str = ".1f") -> None:
    s = _stats(values)
    f = fmt
    print(f"\n  {label}")
    print(f"    n={s['n']}  min={s['min']:{f}}  max={s['max']:{f}}  "
          f"mean={s['mean']:{f}}  std={s['std']:{f}}")
    print(f"    p25={s['p25']:{f}}  p50={s['p50']:{f}}  p75={s['p75']:{f}}  "
          f"p90={s['p90']:{f}}  p95={s['p95']:{f}}  p99={s['p99']:{f}}")
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

    longest_traps  = [r["longest_trap"] for r in results]
    num_traps_lst  = [r["num_traps"]    for r in results]
    num_turns_lst  = [r["num_turns"]    for r in results]
    s_traversal_l  = [r["s_traversal"]  for r in results]
    s_trap_len_l   = [r["s_trap_len"]   for r in results]
    s_trap_cnt_l   = [r["s_trap_cnt"]   for r in results]
    s_turns_l      = [r["s_turns"]      for r in results]
    scores         = [r["score"]        for r in results]

    print("\n" + "=" * 60)
    print("  RAW SIGNAL DISTRIBUTIONS")
    print("=" * 60)
    _print_stats("longest_trap", longest_traps, MAX_TRAP_LEN)
    _print_stats("num_traps",    num_traps_lst,  MAX_NUM_TRAPS)
    _print_stats("num_turns",    num_turns_lst,  MAX_TURNS)

    print("\n" + "=" * 60)
    print("  NORMALISED SIGNAL DISTRIBUTIONS  (all should span [0, 1])")
    print("=" * 60)
    _print_stats("s_traversal  (w=0.25)", s_traversal_l, fmt=".3f")
    _print_stats("s_trap_len   (w=0.25)", s_trap_len_l,  fmt=".3f")
    _print_stats("s_trap_cnt   (w=0.25)", s_trap_cnt_l,  fmt=".3f")
    _print_stats("s_turns      (w=0.25)", s_turns_l,     fmt=".3f")

    print("\n" + "=" * 60)
    print("  FINAL SCORE DISTRIBUTION  (range 0–10)")
    print("=" * 60)
    _print_stats("score", scores, fmt=".3f")

    # Variance summary table
    print("\n" + "=" * 60)
    print("  VARIANCE SUMMARY")
    print("=" * 60)
    print(f"  {'signal':<22}  {'mean':>6}  {'std':>6}  {'var':>8}  {'min':>6}  {'max':>6}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*6}")
    for label, vals in [
        ("s_traversal",  s_traversal_l),
        ("s_trap_len",   s_trap_len_l),
        ("s_trap_cnt",   s_trap_cnt_l),
        ("s_turns",      s_turns_l),
        ("score (/10)",  scores),
    ]:
        st = _stats(vals)
        print(f"  {label:<22}  {st['mean']:>6.3f}  {st['std']:>6.3f}  {st['std']**2:>8.4f}  {st['min']:>6.3f}  {st['max']:>6.3f}")

    print("\n" + "=" * 60)
    print("  SUGGESTED CAPS (p5/p95, rounded to nearest integer)")
    print("=" * 60)
    for label, vals, cur_min, cur_max in (
        ("longest_trap", longest_traps, MIN_TRAP_LEN,  MAX_TRAP_LEN),
        ("num_traps",    num_traps_lst,  MIN_NUM_TRAPS, MAX_NUM_TRAPS),
        ("num_turns",    num_turns_lst,  MIN_TURNS,     MAX_TURNS),
    ):
        p5  = _percentile(vals, 5)
        p95 = _percentile(vals, 95)
        sug_min = int(p5)
        sug_max = max(sug_min + 1, int(p95) + (1 if p95 % 1 else 0))
        m_marker = "" if sug_min == cur_min else f"  ← was {cur_min}"
        M_marker = "" if sug_max == cur_max else f"  ← was {cur_max}"
        print(f"    {label}: MIN={sug_min}{m_marker}  MAX={sug_max}{M_marker}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
