#!/usr/bin/env python3
"""
Open a session pickle in the replay viewer.

Usage:
    python scripts/replay.py sessions/session_0008.pkl
    python scripts/replay.py sessions/session_0008.pkl --tile 24
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.viewer.replay_viewer import load_and_run


def main():
    parser = argparse.ArgumentParser(description="Replay a Pac-Man session")
    parser.add_argument("session", help="Path to session_*.pkl")
    parser.add_argument("--tile", type=int, default=20,
                        help="Tile size in pixels (default: 20)")
    args = parser.parse_args()

    if not os.path.exists(args.session):
        print(f"Error: {args.session} not found", file=sys.stderr)
        sys.exit(1)

    load_and_run(args.session, tile_size=args.tile)


if __name__ == "__main__":
    main()
