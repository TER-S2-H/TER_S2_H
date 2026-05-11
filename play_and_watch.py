#!/usr/bin/env python3
"""
Génère un labyrinthe, joue une partie, sauvegarde le pkl et lance la visualisation.

Usage :
    python play_and_watch.py
    python play_and_watch.py --seed 12345
    python play_and_watch.py --seed 12345 --fps 8
"""

import argparse
import os
import pickle
import random
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.maze_complexity import add_complexity_to_maze_json
from src.pacman_map import PacmanMap
from src.pacman_simulator import (
    load_maze, build_graph, extract_walkable_and_spawns,
    symmetrize_energizers, GameState, step as sim_step,
)
from src.visualize_session import run_viewer


def generate_maze(seed: int, output_dir: str) -> str:
    """Génère un labyrinthe et retourne le chemin vers maze.json."""
    os.makedirs(output_dir, exist_ok=True)
    maze = PacmanMap(seed=seed)
    maze.generate()
    tile_map  = maze.get_tile_map()
    json_path = os.path.join(output_dir, "maze.json")
    maze.export_to_graph_json(json_path, tile_map=tile_map)
    add_complexity_to_maze_json(json_path, start=(13, 23), wrap_tunnels=True)
    print(f"Labyrinthe généré (seed={seed}) → {json_path}")
    return json_path


def run_game(maze_json: str, session_id: int = 0) -> dict:
    """Joue une partie complète et retourne le dict session."""
    random.seed(session_id)

    maze_data = load_maze(maze_json)
    meta   = maze_data.get("metadata", {})
    width  = meta.get("width",  28)
    height = meta.get("height", 31)

    coo_src, coo_dst, node_to_index, index_to_coord, adjacency = build_graph(
        maze_data["nodes"], maze_data["edges"], width, height
    )
    energizers_raw = [
        (e["x"], e["y"]) if isinstance(e, dict) else tuple(e)
        for e in maze_data.get("energizers", [])
    ]
    energizers_full = symmetrize_energizers(energizers_raw, node_to_index, height)
    pacman_spawn, ghost_spawns = extract_walkable_and_spawns(
        maze_data["nodes"], node_to_index, maze_data.get("ghost_walls", set())
    )

    state = GameState(
        adjacency        = adjacency,
        node_to_index    = node_to_index,
        index_to_coord   = index_to_coord,
        ghost_wall_nodes = set(),
        pacman_spawn     = pacman_spawn,
        ghost_spawns     = ghost_spawns,
        initial_pellets = set(
            node_to_index[p]
            for p in maze_data["pellets"]
            if p in node_to_index
        ),
        initial_energizers = set(
            node_to_index[e]
            for e in energizers_full
            if e in node_to_index
        ),
    )

    done, outcome, step_n = False, "unknown", 0
    while not done:
        done, outcome = sim_step(state, is_first_step=(step_n == 0))
        step_n += 1

    print(f"Partie terminée : outcome={outcome}, steps={step_n}, frames={len(state.frames)}")

    return {
        "outcome":    outcome,
        "total_steps": step_n,
        "frames":     state.frames,
    }


def save_session(session: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(session, f)
    print(f"Session sauvegardée → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",    type=int, default=None,
                        help="Seed du labyrinthe (aléatoire si omis)")
    parser.add_argument("--out-dir", default="sessions_watch",
                        help="Dossier de sortie pour les pkl")
    parser.add_argument("--fps",     type=int, default=12,
                        help="Vitesse de lecture (frames/sec)")
    parser.add_argument("--loop",    action="store_true",
                        help="Rejouer en boucle")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 999999)

    # 1. Générer le labyrinthe
    maze_dir  = os.path.join("generated_mazes", f"maze_{seed}")
    maze_json = generate_maze(seed, maze_dir)

    # 2. Jouer la partie
    session = run_game(maze_json)

    # 3. Sauvegarder le pkl
    pkl_path = os.path.join(args.out_dir, f"maze_{seed}", "session_0000.pkl")
    save_session(session, pkl_path)

    # 4. Visualiser
    with open(maze_json) as f:
        maze_data = json.load(f)
    print("Lancement de la visualisation (ESPACE play/pause, Q quitter)...")
    run_viewer(maze_data, session, tile_size=22, fps=args.fps, loop=args.loop)


if __name__ == "__main__":
    main()