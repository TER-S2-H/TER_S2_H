"""
Extrait les métadonnées de jeu des sessions brutes (pacman_simulator.py)
vers des fichiers .npz utilisés pour le calcul des fonctions de coût.

Usage :
    python src/jepacman/to_numpy_meta.py sessions/ jepa_meta/

Structure du npz produit (indices alignés sur les sessions JEPA — dernier frame exclu) :
    pacman_xy    : [n_frames, 2]     int16  — position absolue de Pac-Man
    ghost_xy     : [n_frames, 4, 2]  int16  — positions absolues des 4 fantômes
    n_pellets    : [n_frames]        int16  — nombre de pellets actifs (pellets + energizers)
    outcome_eaten: scalar bool              — True si la session finit sur "eaten"
"""

import os
import glob
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from collections import deque

def bfs_all_pairs(adjacency: dict, n_nodes: int) -> np.ndarray:
    """
    BFS depuis chaque nœud → matrice de distances [n_nodes, n_nodes].
    -1 si inaccessible.
    """
    dist = np.full((n_nodes, n_nodes), -1, dtype=np.int16)
    for src in range(n_nodes):
        dist[src, src] = 0
        queue = deque([src])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency.get(node, []):
                if dist[src, neighbor] == -1:
                    dist[src, neighbor] = dist[src, node] + 1
                    queue.append(neighbor)
    return dist


def convert(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, "**", "*.pkl"), recursive=True))
    if not files:
        print(f"Aucun fichier .pkl trouvé dans {src_dir}")
        return

    for fpath in tqdm(files, desc="Conversion sessions → meta npz"):
        with open(fpath, "rb") as f:
            session = pickle.load(f)

        frames   = session["frames"]
        n_frames = len(frames) - 1
        if n_frames <= 0:
            continue

        # ── Graphe + matrice de distances ─────────────────────────────────
        graph       = session["graph"]
        node_to_idx = {tuple(coord): i
                       for i, coord in graph["index_to_coord"].items()}
        adjacency   = {}
        for src, dst in zip(graph["coo_src"], graph["coo_dst"]):
            adjacency.setdefault(src, []).append(dst)

        dist_matrix = bfs_all_pairs(adjacency, graph["num_nodes"])

        # ── Arrays par frame ───────────────────────────────────────────────
        pacman_xy  = np.zeros((n_frames, 2),    dtype=np.int16)
        ghost_xy   = np.zeros((n_frames, 4, 2), dtype=np.int16)
        n_pellets  = np.zeros(n_frames,          dtype=np.int16)
        ghost_dist = np.zeros((n_frames, 4),     dtype=np.int16)

        for t in range(n_frames):
            frame = frames[t]

            px, py = frame["pacman_pos"]
            pacman_xy[t] = [px, py]
            pac_idx = node_to_idx.get((px, py), -1)

            for g in range(4):
                gx, gy = frame["ghost_positions"][g]
                ghost_xy[t, g] = [gx, gy]
                g_idx = node_to_idx.get((gx, gy), -1)
                if pac_idx >= 0 and g_idx >= 0:
                    d = dist_matrix[pac_idx, g_idx]
                    ghost_dist[t, g] = max(0, d)

            n_pellets[t] = (
                len(frame["active_pellets"]) + len(frame["active_energizers"])
            )

        outcome_eaten = np.array(
            session.get("outcome", "") == "eaten", dtype=bool
        )

        rel = os.path.relpath(fpath, src_dir)
        out = os.path.join(dst_dir, rel).replace(".pkl", ".npz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(
            out,
            pacman_xy=pacman_xy,
            ghost_xy=ghost_xy,
            n_pellets=n_pellets,
            outcome_eaten=outcome_eaten,
            ghost_dist_to_pacman=ghost_dist,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src", help="Répertoire des sessions brutes (.pkl)")
    p.add_argument("dst", help="Répertoire de sortie pour les meta .npz")
    args = p.parse_args()
    convert(args.src, args.dst)