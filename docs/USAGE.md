# Usage Guide

## Quick Start

### Prerequisites

```bash
pip install pygame  # for the replay viewer
```

All scripts assume you are at the project root (`TER_S2_H/`).

---

## Ghost Agent: `iceguct`

`iceguct` is the collaborative ghost-team agent based on ICE gUCT.  It uses a
ghost-only junction graph plus a KNN Pac-Man predictor loaded from
`data/knn_train/dataset.npz`.

Construct it through the registry:

```python
from src.agents import get_agent

ghosts = get_agent("iceguct")()
```

Run it from the CLI like any other registered ghost agent:

```bash
python scripts/play.py \
    --maze generated_maze/maze_858558/maze.json \
    --pacman mcts_pacman \
    --ghosts iceguct
```

### KNN dataset

The predictor expects `data/knn_train/dataset.npz` with two arrays:

- `X` with shape `(N, 10)` for the feature vectors.
- `Y` with shape `(N, 10)` for the move vectors.

If the file is missing or empty, `iceguct` logs a warning and falls back to
random legal Pac-Man prediction during rollouts.  That keeps the agent usable
for bootstrap runs before the dataset exists.

### Collecting trajectories

Use the collection script to generate or refresh the dataset:

```bash
python scripts/collect_trajectories.py \
    --maze-dir generated_maze \
    --pacman heuristic \
    --ghosts random \
    --n-games 100 \
    --seed 42 \
    --out data/knn_train/
```

The script writes:

- `session_<id>.pkl` files for each game.
- `dataset.npz` with the concatenated `X` and `Y` arrays.

### Configuration knobs

The agent is configured through `IceGuctConfig`.

```python
from src.agents.iceguct_ghosts import IceGuctConfig, IceGuctGhosts

cfg = IceGuctConfig(
    C=1.41,
    epsilon=1e-6,
    T_sim=80,
    k_neighbors=3,
    p_predict_start=0.9,
    p_predict_end=0.1,
    budget_ms_per_tick=40,
)
ghosts = IceGuctGhosts(config=cfg)
```

Useful tuning levers:

- `budget_ms_per_tick` controls the total wall-clock budget per tick.
- `T_sim` controls the rollout horizon.
- `k_neighbors` controls the KNN smoothing.
- `p_predict_start` and `p_predict_end` control how aggressively rollouts trust the predictor.

### Troubleshooting

- Empty dataset warning: regenerate `data/knn_train/dataset.npz` with the collection script.
- Slow `act()`: lower `budget_ms_per_tick` or `T_sim`.
- Unstable predictor output: refresh the dataset with more heuristic-Pac-Man trajectories.

---

## Playing a Game

Run MCTS Pac-Man against heuristic ghosts on the included maze:

```bash
python scripts/play.py \
    --maze generated_maze/maze_858558/maze.json \
    --pacman mcts_pacman \
    --ghosts random_ghosts
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--maze` | *(required)* | Path to `maze.json` |
| `--pacman` | `mcts_pacman` | Agent name: `mcts_pacman`, `heuristic_pacman`, `random_pacman` |
| `--ghosts` | `heuristic_ghosts` | Agent name: `random_ghosts`, `heuristic_ghosts`, `iceguct` |
| `--seed` | `0` | Random seed for reproducibility |
| `--max-steps` | `5000` | Step cap before timeout |
| `--out` | *(none)* | Save session pickle to this path |

---

## Replaying a Session

```bash
python scripts/replay.py sessions/session_0008.pkl
python scripts/replay.py sessions/session_0008.pkl --tile 24
```

**Keyboard controls in the viewer:**

| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `→` (right arrow) | Step forward one frame |
| `←` (left arrow) | Step backward one frame |
| `H` | Toggle heatmap overlay (visit frequency) |
| `Q` or `Esc` | Quit |

---

## Running a Benchmark

Evaluate one or more agent pairings over a directory of mazes:

```bash
python scripts/benchmark.py \
    --maze-dir generated_maze \
    --n-games 20 \
    --out sessions/
```

The benchmark prints a JSON summary with win rates and 95 % confidence intervals.
If you omit `--pacman` or `--ghosts`, the script benchmarks every registered agent for that role. You can also pass `all` explicitly, for example `--pacman all --ghosts all`.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--maze-dir` | `generated_maze` | Directory containing `maze.json` files (searched recursively) |
| `--pacman` | *(all registered Pac-Man agents by omission)* | One or more Pac-Man agent names |
| `--ghosts` | *(all registered ghost agents by omission)* | One or more ghost agent names |
| `--n-games` | `5` | Games per (pacman, ghosts, maze) triple |
| `--out` | *(none)* | Directory for session pickles + `summary.json` |
| `--no-parallel` | *(flag)* | Disable parallel execution |
| `--max-steps` | `5000` | Per-game step cap |
| `--verbose` | *(flag)* | Enable DEBUG logging |

---

## Agent Registry

All agents are registered in `src/agents/__init__.py`.  To use an agent by
name, pass it to `get_agent(name)`:

```python
from src.agents import get_agent
PacmanClass = get_agent("mcts_pacman")
agent = PacmanClass()
```

Available names:

| Name | Class | Role |
|------|-------|------|
| `mcts_pacman` | `MCTSPacmanAgent` | pacman |
| `heuristic_pacman` | `HeuristicPacman` | pacman |
| `random_pacman` | `RandomPacmanAgent` | pacman |
| `iceguct` | `IceGuctGhosts` | ghosts |
| `heuristic_ghosts` | `HeuristicGhosts` | ghosts |
| `random_ghosts` | `RandomGhostsAgent` | ghosts |

---

## Tuning MCTS Parameters

The MCTS Pac-Man agent is configured via `PacmanMCTSConfig`:

```python
from src.agents.mcts_pacman import MCTSPacmanAgent, PacmanMCTSConfig

cfg = PacmanMCTSConfig(
    n_iterations = 60,    # iterations per junction decision
    T_time       = 40,    # rollout horizon (ticks)
    T_path       = 60,    # max corridor depth in tree (cells)
    K_endgame    = 40,    # pellet threshold for endgame tactic
    T_survival   = 0.7,   # survival threshold for tactic switch
    C            = 1.5,   # UCT exploration constant
)
agent = MCTSPacmanAgent(cfg)
```

**Performance vs quality trade-off:**

| `n_iterations` | `T_time` | ms/step (approx) | Win rate vs random ghosts |
|----------------|----------|------------------|---------------------------|
| 60 | 40 | ~13 ms | 85 % (17/20) |
| 100 | 60 | ~40 ms | 85 %+ |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests cover:
- `test_full_reachability.py` — every PATH cell reachable from spawn via `pacman_graph`
- `test_ghost_house_excluded.py` — no ghost-house cell in `pacman_graph`
- `test_simulator.py` — 5 simulator unit tests
- `test_coverage_smoke.py` — MCTS vs random ghosts visits all reachable cells
- `test_ghost_graph_includes_ghost_house.py` — ghost graph includes the ghost house and differs only there
- `test_blinky_chase.py` / `test_blinky_flee_when_frightened.py` — Blinky chase/flee behavior
- `test_knn_cold_start.py` / `test_knn_prediction_shape.py` — KNN predictor fallbacks and speed
- `test_tree_rebuild_on_global_reverse.py` — tree rebuild after energizer-triggered reverse
- `test_iceguct_compute_budget.py` — act() respects the configured wall-clock budget
- `test_iceguct_vs_random_pacman.py` — smoke test against a random Pac-Man
- `test_data_collection_script.py` — trajectory collection writes sessions and dataset.npz

---

## Session Schema

Session pickles are plain Python dicts with the following keys:

```python
{
    "graph": {
        "coo_src": [...],        # COO adjacency (source indices)
        "coo_dst": [...],        # COO adjacency (destination indices)
        "node_to_index": {...},  # "x,y" -> int
        "index_to_coord": {...}, # str(int) -> [x, y]
        "ghost_wall_nodes": [...],
        "num_nodes": int,
    },
    "initial_pellets":    [(x, y), ...],
    "initial_energizers": [(x, y), ...],
    "pacman_spawn":       (x, y),
    "ghost_spawns":       [(x, y), (x, y), (x, y), (x, y)],
    "outcome":            "cleared" | "eaten" | "timeout",
    "total_steps":        int,
    "frames": [
        {
            "step": int,
            "pacman_pos": (x, y),
            "ghost_positions": [(x,y), ...],   # 4 ghosts
            "ghost_frightened": [int, ...],     # remaining frightened ticks
            "active_pellets": [(x,y), ...],
            "active_energizers": [(x,y), ...],
        },
        ...
    ],
}
```
