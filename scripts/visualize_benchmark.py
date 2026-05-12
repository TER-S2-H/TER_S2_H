#!/usr/bin/env python3
"""
Generate a PNG benchmark report from a summary.json.

Three heatmaps: avg steps, pellet-eaten rate, win rate.

Usage:
    python scripts/visualize_benchmark.py sessions/summary.json
    python scripts/visualize_benchmark.py sessions/summary.json --out report.png
"""

import argparse
import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Human-readable agent labels
_LABELS = {
    "heuristic_pacman":  "Heuristique",
    "mcts_pacman":       "MCTS",
    "random_pacman":     "Aléatoire",
    "heuristic_ghosts":  "Heuristique",
    "mcts_ghosts":       "MCTS",
    "iceguct":           "ICEgUCT",
    "random_ghosts":     "Aléatoire",
}


def _label(name: str) -> str:
    return _LABELS.get(name, name)


def _aggregate(results):
    pac_agents   = sorted({r["pacman"] for r in results})
    ghost_agents = sorted({r["ghosts"] for r in results})
    buckets = collections.defaultdict(list)
    for r in results:
        buckets[(r["pacman"], r["ghosts"])].append(r)

    agg = {}
    for p in pac_agents:
        agg[p] = {}
        for g in ghost_agents:
            rs = buckets[(p, g)]
            if not rs:
                agg[p][g] = None
                continue
            n = len(rs)
            agg[p][g] = {
                "wr":        sum(r["cleared"] for r in rs) / n,
                "pell_frac": sum(r["pellets_eaten"] / max(r["pellets_total"], 1)
                                 for r in rs) / n,
                "avg_steps": sum(r["steps"] for r in rs) / n,
            }
    return agg, pac_agents, ghost_agents


def _matrix(agg, pacs, ghosts, key):
    mat = np.full((len(pacs), len(ghosts)), np.nan)
    for i, p in enumerate(pacs):
        for j, g in enumerate(ghosts):
            v = agg[p][g]
            if v is not None:
                mat[i, j] = v[key]
    return mat


def _draw_heatmap(ax, mat, row_labels, col_labels, title, fmt, cmap, vmin, vmax):
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("Fantômes", fontsize=11)
    ax.set_ylabel("Pac-Man", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                continue
            norm = (val - vmin) / max(vmax - vmin, 1e-9)
            # RdYlGn: high norm = green (bright), low norm = red (dark)
            # choose text contrast based on luminance of cell
            color = "white" if norm < 0.25 or norm > 0.85 else "black"
            ax.text(j, i, fmt(val), ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", help="Path to summary.json")
    parser.add_argument("--out", default=None)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    with open(args.summary) as f:
        data = json.load(f)

    results = [
        r for r in data["results"]
        if "jepa" not in r["pacman"] and "jepa" not in r["ghosts"]
        and "random" not in r["pacman"] and "random" not in r["ghosts"]
    ]

    agg, pacs, ghosts = _aggregate(results)
    pac_labels   = [_label(p) for p in pacs]
    ghost_labels = [_label(g) for g in ghosts]

    mat_steps = _matrix(agg, pacs, ghosts, "avg_steps")
    mat_pell  = _matrix(agg, pacs, ghosts, "pell_frac")
    mat_wr    = _matrix(agg, pacs, ghosts, "wr")

    n_pac, n_ghost = len(pacs), len(ghosts)
    fig, axes = plt.subplots(1, 3, figsize=(4.5 * n_ghost, 2.2 + 0.9 * n_pac))
    fig.suptitle(
        "Benchmark : Pac-Man vs Fantômes",
        fontsize=14, fontweight="bold",
    )

    # Durée : low=green, high=red  →  RdYlGn_r  (matches Figure_1)
    s_min, s_max = np.nanmin(mat_steps), np.nanmax(mat_steps)
    im0 = _draw_heatmap(axes[0], mat_steps, pac_labels, ghost_labels,
                        "Durée (steps)",
                        lambda v: f"{v:.0f}",
                        "RdYlGn", s_min, s_max)
    cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.ax.tick_params(labelsize=9)

    # Pellets : low=red, high=green  →  RdYlGn
    im1 = _draw_heatmap(axes[1], mat_pell, pac_labels, ghost_labels,
                        "Taux de pellets mangés",
                        lambda v: f"{v*100:.0f}%",
                        "RdYlGn", 0.0, 1.0)
    cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    cb1.ax.tick_params(labelsize=9)

    # Win rate : low=red, high=green  →  RdYlGn
    im2 = _draw_heatmap(axes[2], mat_wr, pac_labels, ghost_labels,
                        "Taux de victoire Pac-Man",
                        lambda v: f"{v*100:.0f}%",
                        "RdYlGn", 0.0, 1.0)
    cb2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cb2.ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    cb2.ax.tick_params(labelsize=9)

    fig.tight_layout()

    out_path = args.out or os.path.join(os.path.dirname(args.summary), "report.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
