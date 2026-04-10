#!/usr/bin/env python3
"""Inspect Pac-Man simulator session pickle files."""

import argparse
import os
import pickle
from collections import Counter
from statistics import mean
from typing import Dict, List


def load_session(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def print_single_session(path: str, show_frames: bool) -> None:
    session = load_session(path)

    print(f"File: {path}")
    print(f"Top-level keys: {sorted(session.keys())}")
    print(f"Outcome: {session['outcome']}")
    print(f"Total steps: {session['total_steps']}")
    print(f"Frames: {len(session['frames'])}")
    print(f"Pac-Man spawn: {session['pacman_spawn']}")
    print(f"Ghost spawns: {session['ghost_spawns']}")

    graph = session.get("graph", {})
    print(f"Graph nodes: {graph.get('num_nodes')}")
    print(f"Graph directed edges: {len(graph.get('coo_src', []))}")

    if show_frames and session.get("frames"):
        print("\nFirst frame:")
        print(session["frames"][0])
        print("\nLast frame:")
        print(session["frames"][-1])


def list_pickle_files(folder: str) -> List[str]:
    files = [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if name.endswith(".pkl")
    ]
    return files


def print_directory_summary(folder: str) -> None:
    files = list_pickle_files(folder)
    if not files:
        print(f"No .pkl files found in: {folder}")
        return

    outcomes = Counter()
    steps: List[int] = []

    for path in files:
        session = load_session(path)
        outcomes[session.get("outcome", "unknown")] += 1
        steps.append(int(session.get("total_steps", 0)))

    print(f"Folder: {folder}")
    print(f"Session files: {len(files)}")
    print(f"Outcome distribution: {dict(outcomes)}")
    print(
        "Steps stats: "
        f"min={min(steps)} "
        f"max={max(steps)} "
        f"mean={mean(steps):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read Pac-Man simulator .pkl session files"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to one session .pkl file")
    group.add_argument("--dir", help="Folder containing session .pkl files")

    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="Do not print first/last frame when using --file",
    )

    args = parser.parse_args()

    if args.file:
        print_single_session(args.file, show_frames=not args.no_frames)
    else:
        print_directory_summary(args.dir)


if __name__ == "__main__":
    main()
