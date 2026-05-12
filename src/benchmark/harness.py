"""
Benchmark harness: run multiple games across multiple agent pairings and mazes.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..game.maze import Maze
from .run_game import run_game, save_session

logger = logging.getLogger(__name__)


@dataclass
class PairingResult:
    pacman_name: str
    ghost_name:  str
    maze_name:   str

    outcome:         str
    total_steps:     int
    pellets_eaten:   int
    pellets_total:   int
    ghosts_eaten:    int
    cleared:         bool


@dataclass
class BenchmarkReport:
    results: List[PairingResult] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        if not self.results:
            return {}

        n = len(self.results)
        win_rate    = sum(r.cleared for r in self.results) / n
        avg_steps   = sum(r.total_steps for r in self.results) / n
        avg_pellets = sum(r.pellets_eaten / max(r.pellets_total, 1) for r in self.results) / n
        avg_ghosts  = sum(r.ghosts_eaten for r in self.results) / n

        # 95% CI for win_rate (Wald interval)
        se = math.sqrt(win_rate * (1 - win_rate) / n) if n > 1 else 0.0
        ci = 1.96 * se

        return {
            "n_games":      n,
            "win_rate":     round(win_rate, 4),
            "win_rate_ci95": round(ci, 4),
            "avg_steps":    round(avg_steps, 1),
            "avg_pellets_frac": round(avg_pellets, 4),
            "avg_ghosts_eaten": round(avg_ghosts, 2),
            "outcomes":     {
                o: sum(r.outcome == o for r in self.results)
                for o in ("eaten", "cleared", "timeout")
            },
        }

    def to_json(self) -> dict:
        return {
            "summary": self.summary(),
            "results": [
                {
                    "pacman": r.pacman_name,
                    "ghosts": r.ghost_name,
                    "maze":   r.maze_name,
                    "outcome": r.outcome,
                    "steps":   r.total_steps,
                    "pellets_eaten": r.pellets_eaten,
                    "pellets_total": r.pellets_total,
                    "ghosts_eaten":  r.ghosts_eaten,
                    "cleared": r.cleared,
                }
                for r in self.results
            ],
        }


def _run_one(
    maze_path: str,
    pacman_name: str,
    ghost_name:  str,
    seed: int,
    out_dir: Optional[str],
    game_idx: int,
    max_steps: int,
) -> PairingResult:
    # Import inside worker so factories are resolved after fork (avoids pickle)
    from ..agents import get_agent
    maze    = Maze.load(maze_path)
    pac_agt = get_agent(pacman_name)()
    gst_agt = get_agent(ghost_name)()

    record = run_game(
        maze,
        pac_agt,
        gst_agt,
        seed=seed,
        max_steps=max_steps,
        record=(out_dir is not None),
    )

    if out_dir is not None:
        fname = os.path.join(out_dir, f"session_{game_idx:04d}.pkl")
        save_session(record, fname)

    maze_name   = os.path.splitext(os.path.basename(maze_path))[0]
    total       = len(record["initial_pellets"]) + len(record["initial_energizers"])
    remaining   = (len(record["frames"][-1]["active_pellets"]) +
                   len(record["frames"][-1]["active_energizers"])) if record["frames"] else 0
    eaten       = total - remaining

    # Count ghosts eaten
    ghosts_eaten = 0
    for i in range(len(record["frames"]) - 1):
        f_prev = record["frames"][i]
        f_next = record["frames"][i + 1]
        for j in range(4):
            if (f_prev["ghost_frightened"][j] > 0 and
                f_next["ghost_frightened"][j] == 0 and
                tuple(f_next["ghost_positions"][j]) in [tuple(s) for s in record["ghost_spawns"]]):
                ghosts_eaten += 1

    return PairingResult(
        pacman_name    = pacman_name,
        ghost_name     = ghost_name,
        maze_name      = maze_name,
        outcome        = record["outcome"],
        total_steps    = record["total_steps"],
        pellets_eaten  = eaten,
        pellets_total  = total,
        ghosts_eaten   = ghosts_eaten,
        cleared        = record["outcome"] == "cleared",
    )


def benchmark(
    maze_dir: str,
    pacman_agents: Dict[str, Any],   # only keys (names) are used
    ghost_agents:  Dict[str, Any],
    n_games_per_pairing: int,
    out_dir: Optional[str] = None,
    parallel: bool = True,
    max_steps: int = 5000,
) -> BenchmarkReport:
    """
    Run n_games_per_pairing games for each (pacman, ghost, maze) triple.
    Writes session pickles to out_dir (if given) and a summary.json.
    """
    maze_paths = sorted(glob.glob(os.path.join(maze_dir, "**", "maze.json"), recursive=True))
    if not maze_paths:
        maze_paths = sorted(glob.glob(os.path.join(maze_dir, "*.json")))
    if not maze_paths:
        raise FileNotFoundError(f"No maze*.json files found under {maze_dir}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tasks = []
    game_idx = 0
    for maze_path in maze_paths:
        for p_name in pacman_agents:
            for g_name in ghost_agents:
                for game_i in range(n_games_per_pairing):
                    seed = hash((maze_path, p_name, g_name, game_i)) & 0xFFFFFFFF
                    tasks.append((maze_path, p_name, g_name,
                                  seed, out_dir, game_idx, max_steps))
                    game_idx += 1

    report = BenchmarkReport()

    if parallel and len(tasks) > 1:
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(_run_one, *t): t for t in tasks
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    report.results.append(result)
                    logger.info(
                        "%s vs %s on %s: %s (%d steps)",
                        result.pacman_name, result.ghost_name, result.maze_name,
                        result.outcome, result.total_steps,
                    )
                except Exception as exc:
                    logger.error("Game failed: %s", exc)
    else:
        for t in tasks:
            try:
                result = _run_one(*t)
                report.results.append(result)
                logger.info(
                    "%s vs %s on %s: %s (%d steps)",
                    result.pacman_name, result.ghost_name, result.maze_name,
                    result.outcome, result.total_steps,
                )
            except Exception as exc:
                logger.error("Game failed: %s", exc)

    if out_dir:
        summary_path = os.path.join(out_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(report.to_json(), f, indent=2)
        logger.info("Summary written to %s", summary_path)

    return report
