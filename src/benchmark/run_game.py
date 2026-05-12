"""
run_game: play one game and optionally record a session pickle.

The SessionRecord schema exactly matches §2 of MCTS_AGENTS_TASK_BRIEF.md
so it is byte-compatible with the JEPA harness.
"""

from __future__ import annotations

import logging
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple

from ..agents.base import Agent
from ..game.maze import Maze
from ..game.simulator import SimState, initial_state, step as sim_step
from ..game.types import Direction, GhostAction, Observation, PacmanAction

logger = logging.getLogger(__name__)


# Type alias for the session dict (matches §2 schema exactly)
SessionRecord = Dict[str, Any]


def run_game(
    maze: Maze,
    pacman_agent: Agent,
    ghost_agent: Agent,
    *,
    seed: Optional[int] = None,
    max_steps: int = 5000,
    record: bool = True,
) -> SessionRecord:
    """
    Play one game.  Returns a SessionRecord matching the §2 schema.

    Parameters
    ----------
    maze         : loaded Maze instance
    pacman_agent : Agent with role='pacman'
    ghost_agent  : Agent with role='ghosts'
    seed         : RNG seed for reproducibility
    max_steps    : safety cap (outcome = 'timeout')
    record       : if False skip frame list (faster benchmarking)
    """
    if seed is not None:
        random.seed(seed)

    pacman_agent.reset(maze, "pacman")
    ghost_agent.reset(maze, "ghosts")

    state = initial_state(maze)

    # ---- Build graph dict for the session pickle ----------------------
    nodes_sorted = sorted(maze.pacman_graph.keys())
    node_to_index   = {n: i for i, n in enumerate(nodes_sorted)}
    index_to_coord  = {i: n for i, n in enumerate(nodes_sorted)}
    coo_src, coo_dst = [], []
    for n, nbrs in maze.pacman_graph.items():
        for nb in nbrs:
            coo_src.append(node_to_index[n])
            coo_dst.append(node_to_index[nb])

    graph_dict = {
        "coo_src":       coo_src,
        "coo_dst":       coo_dst,
        "node_to_index": {f"{n[0]},{n[1]}": i for n, i in node_to_index.items()},
        "index_to_coord": {str(i): list(c) for i, c in index_to_coord.items()},
        "ghost_wall_nodes": sorted(maze.ghost_house),
        "num_nodes": len(nodes_sorted),
    }

    frames: List[dict] = []

    if record:
        frames.append(state.to_observation())

    while not state.is_terminal:
        obs = Observation(
            step              = state.step,
            pacman_pos        = state.pacman_pos,
            ghost_positions   = state.ghost_positions,
            ghost_frightened  = state.ghost_frightened,
            active_pellets    = list(state.active_pellets),
            active_energizers = list(state.active_energizers),
        )

        pac_action: PacmanAction = pacman_agent.act(obs)
        gst_action: GhostAction  = ghost_agent.act(obs)

        state = sim_step(
            state, maze, pac_action, gst_action,
            frightened_duration=40,
            max_steps=max_steps,
        )

        if record:
            frames.append(state.to_observation())

        if state.step % 500 == 0:
            logger.info(
                "step=%d pellets=%d energizers=%d outcome=%s",
                state.step,
                len(state.active_pellets),
                len(state.active_energizers),
                state.outcome,
            )

    outcome = state.outcome or "timeout"
    logger.info(
        "Game over: outcome=%s steps=%d pellets_remaining=%d",
        outcome, state.step,
        len(state.active_pellets) + len(state.active_energizers),
    )

    return {
        "graph":             graph_dict,
        "initial_pellets":   sorted(maze.initial_pellets),
        "initial_energizers": sorted(maze.initial_energizers),
        "pacman_spawn":      maze.pacman_spawn,
        "ghost_spawns":      maze.ghost_spawns,
        "outcome":           outcome,
        "total_steps":       state.step,
        "frames":            frames,
    }


def save_session(record: SessionRecord, path: str):
    with open(path, "wb") as f:
        pickle.dump(record, f)


def load_session(path: str) -> SessionRecord:
    with open(path, "rb") as f:
        return pickle.load(f)
