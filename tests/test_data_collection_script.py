"""The trajectory-collection script should write sessions and a well-shaped dataset."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_data_collection_script(tmp_path):
    out_dir = tmp_path / "knn_train"
    script = Path(__file__).resolve().parents[1] / "scripts" / "collect_trajectories.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--maze-dir",
            "generated_maze",
            "--pacman",
            "heuristic",
            "--ghosts",
            "random",
            "--n-games",
            "2",
            "--seed",
            "7",
            "--out",
            str(out_dir),
            "--max-steps",
            "50",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    sessions = sorted(out_dir.glob("session_*.pkl"))
    assert len(sessions) == 2

    dataset = np.load(out_dir / "dataset.npz")
    assert dataset["X"].ndim == 2
    assert dataset["Y"].ndim == 2
    assert dataset["X"].shape == dataset["Y"].shape
    assert dataset["X"].shape[1] == 10
    assert dataset["X"].shape[0] > 0
