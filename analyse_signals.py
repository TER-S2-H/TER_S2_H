#!/usr/bin/env python3
"""
Statistical analysis of maze difficulty signals.

Generates N mazes in-memory, computes complexity signals for each, and
reports the full distribution of raw signals and normalised tanh-z signals
so that MEAN_* / STD_* constants in maze_complexity.py can be recalibrated.

Usage:
    python3 analyse_signals.py [--n N] [--seed SEED] [--start-seed S]
"""

import argparse
import math
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
    TRAP_MIN_INTERIOR,
    MEAN_STEPS_RATIO, STD_STEPS_RATIO,
    MEAN_TRAP_LEN, STD_TRAP_LEN,
    MEAN_TRAP_CNT, STD_TRAP_CNT,
    W_STEPS, W_TRAP_LEN, W_TRAP_CNT,
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
        trap_corridors, longest_trap, num_traps = find_trap_corridors(
            pos_list, adjacency, min_interior=TRAP_MIN_INTERIOR
        )
        avg_trap_len = (
            sum(len(c) for c in trap_corridors) / num_traps if num_traps > 0 else 0.0
        )

        # Normalized signals — tanh-z (same formula as compute_complexity_score)
        steps_ratio = steps / food_count if food_count > 0 else MEAN_STEPS_RATIO
        s_steps    = math.tanh((steps_ratio  - MEAN_STEPS_RATIO) / STD_STEPS_RATIO)
        s_trap_len = math.tanh((longest_trap - MEAN_TRAP_LEN)    / STD_TRAP_LEN)
        s_trap_cnt = math.tanh((num_traps    - MEAN_TRAP_CNT)    / STD_TRAP_CNT)
        combined = W_STEPS * s_steps + W_TRAP_LEN * s_trap_len + W_TRAP_CNT * s_trap_cnt
        score = (combined + 1.0) / 2.0 * 10.0

        return {
            "seed":         seed,
            "food_count":   food_count,
            "steps":        steps,
            "longest_trap": longest_trap,
            "avg_trap_len": avg_trap_len,
            "num_traps":    num_traps,
            "num_turns":    num_turns,
            "s_steps":      s_steps,
            "s_trap_len":   s_trap_len,
            "s_trap_cnt":   s_trap_cnt,
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
    avg_trap_lens  = [r["avg_trap_len"] for r in results]
    num_traps_lst  = [r["num_traps"]    for r in results]
    num_turns_lst  = [r["num_turns"]    for r in results]
    s_steps_l      = [r["s_steps"]      for r in results]
    s_trap_len_l   = [r["s_trap_len"]   for r in results]
    s_trap_cnt_l   = [r["s_trap_cnt"]   for r in results]
    scores         = [r["score"]        for r in results]

    print("\n" + "=" * 60)
    print("  RAW SIGNAL DISTRIBUTIONS")
    print("=" * 60)
    _print_stats("longest_trap", longest_traps)
    _print_stats("avg_trap_len", avg_trap_lens)
    _print_stats("num_traps",    num_traps_lst)
    _print_stats("num_turns",    num_turns_lst)

    print("\n" + "=" * 60)
    print("  NORMALISED SIGNAL DISTRIBUTIONS  (tanh-z, range ≈ (-1, +1))")
    print("=" * 60)
    _print_stats("s_steps      (w=1/3)", s_steps_l,    fmt=".3f")
    _print_stats("s_trap_len   (w=1/3)", s_trap_len_l, fmt=".3f")
    _print_stats("s_trap_cnt   (w=1/3)", s_trap_cnt_l, fmt=".3f")

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
        ("s_steps",      s_steps_l),
        ("s_trap_len",   s_trap_len_l),
        ("s_trap_cnt",   s_trap_cnt_l),
        ("score (/10)",  scores),
    ]:
        st = _stats(vals)
        print(f"  {label:<22}  {st['mean']:>6.3f}  {st['std']:>6.3f}  {st['std']**2:>8.4f}  {st['min']:>6.3f}  {st['max']:>6.3f}")

    print("\n" + "=" * 60)
    print("  SUGGESTED MEAN/STD RECALIBRATION (from this run)")
    print("=" * 60)
    steps_ratios = [r["steps"] / r["food_count"] for r in results if r["food_count"] > 0]
    for label, vals, cur_mean, cur_std in (
        ("steps_ratio",  steps_ratios,  MEAN_STEPS_RATIO, STD_STEPS_RATIO),
        ("longest_trap", longest_traps, MEAN_TRAP_LEN,    STD_TRAP_LEN),
        ("num_traps",    num_traps_lst, MEAN_TRAP_CNT,    STD_TRAP_CNT),
    ):
        st = _stats(vals)
        m_marker = "" if abs(st["mean"] - cur_mean) < 0.01 else f"  ← was {cur_mean:.5f}"
        s_marker = "" if abs(st["std"]  - cur_std)  < 0.01 else f"  ← was {cur_std:.5f}"
        print(f"    {label}: MEAN={st['mean']:.5f}{m_marker}  STD={st['std']:.5f}{s_marker}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
