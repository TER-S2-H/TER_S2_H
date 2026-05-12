# Task Brief: ICE gUCT Ghost-Team Agent

## 1. Goal and scope

Replace the existing ghost-team agent (a joint-action MCTS) with a new ghost-team agent based on the ICE gUCT approach from Nguyen & Thawonmas, *"Applying Monte-Carlo Tree Search to Collaboratively Controlling of a Ghost Team in Ms Pac-Man"* (IGIC 2011, winner of the CEC 2011 Ms Pac-Man vs Ghosts ghost-side competition). The paper PDF is provided.

**Strictly in scope:** the ghost-team agent, a KNN-based Pac-Man movement predictor, a trajectory-data collection script, and tests + docs for the above.

**Strictly out of scope:** the existing Pac-Man agent, the game simulator, the maze loader, the replay viewer, the session/observation/action types, and the benchmark harness. All of these already exist in the codebase. **Read them first. Do not modify them** unless the `Agent` protocol or a discovered bug requires it (in which case ask the user first).

**Discard the old ghost agent.** Delete the existing ghost-team agent module(s) entirely once the new one is in place and tests pass. Do not try to preserve any of the old joint-action tree logic — the design is fundamentally different.

---

## 2. First steps before coding

1. Read `igic11.pdf` end-to-end.
2. Read the codebase to find:
   - The `Agent` protocol (the new ghost agent must implement it).
   - The game simulator / state types — especially how it represents Pac-Man's direction, ghost positions, the `frightened` per-ghost timer, energizer events, and the per-move compute budget.
   - The maze loader and any existing junction/C-road graph utilities. If the Pac-Man agent already builds a junction graph, reuse it. **Do not duplicate that work.**
   - The existing heuristic Pac-Man (rollout policy from the Pepels paper, usually exposed as a standalone agent). You need it for KNN data collection.
   - Where benchmark sessions are written and what the registry / agent factory looks like for adding agents.
3. Confirm with the user before writing code:
   - **Q1.** Does the game force a "global reverse" (all ghosts flip direction) the tick Pac-Man eats an energizer? (Original Ms Pac-Man does. ICE gUCT relies on this as a tree-rebuild trigger.) If no, propose using the energizer-eaten event itself as the trigger.
   - **Q2.** Do frightened ghosts move at half speed in the simulator? (Original Ms Pac-Man does. Affects rollout fidelity.)
   - **Q3.** Do ghosts decide every tick or only at junctions in the existing simulator? Affects whether per-ghost trees rebuild on every junction visit (paper assumes junction-only).
   - **Q4.** What is the per-move compute budget (wall-clock) for the ghost agent? The paper used 40 ms total. The new agent splits this across 3 MCTS ghosts (either sequentially or in parallel — see §3.7).

Do not start coding until Q1–Q3 are answered. Q4 has a sensible default (whatever budget the existing agent uses).

---

## 3. Algorithm: ICE gUCT adapted

### 3.1 Roles

Four ghosts. **Three run MCTS** with one tree each; call them G1, G2, G3. **One runs a rule** ("Blinky" role). Assign the role to whichever ghost index the simulator uses as Blinky (usually index 0). If the simulator has no role distinction, the assignment is arbitrary but fixed for the duration of a game.

### 3.2 Graphs

Use the existing junction graph if available, otherwise build it once at maze load:
- **Ghost adjacency graph** — every walkable cell *including* `ghost_walls`, since ghosts can pass through the ghost house.
- **Junction graph** — nodes are cross-points (cells with ≥3 neighbors) and dead-ends; edges are C-roads (corridor sequences) with their length in cells as weight.
- Precompute all-pairs shortest-path distances on the junction graph.

The Pac-Man agent already has an analogous graph that *excludes* `ghost_walls`. The ghost agent must use the inclusive one. **Verify in a unit test that the two graphs differ exactly on the ghost-house subgraph.**

### 3.3 Tree structure (per MCTS ghost)

- **Root** = the next cross-point the ghost is about to reach (not its current cell). If the ghost is already at a cross-point, root is that cross-point.
- **Children** = adjacent cross-points reachable along outgoing C-roads, **excluding the corridor the ghost arrived from** (no reversing — matches game physics).
- Each tree node stores: visit count `T_i`, accumulated reward `X_i`.

UCT/UCB1 formula for selection (paper eq. 1):
```
UCB1(i) = X_i / T_i + C · sqrt( ln(T) / (T_i + ε) )
```
with `C` configurable (start at √2 ≈ 1.41, paper doesn't specify a value), and `ε ≈ 1e-6`.

### 3.4 The Blinky rule

When it is Blinky's turn to choose a direction at a cross-point:
1. Infer Pac-Man's current heading from the diff of the last two known `pacman_pos` values (or from whatever the simulator exposes).
2. Identify the next cross-point Pac-Man is heading toward along that corridor.
3. BFS on the **ghost adjacency graph** to find the shortest path from Blinky's current cell to that cross-point, and pick the first move along it.

If Pac-Man's heading is ambiguous (just spawned, or just reversed), fall back to BFS-shortest-path to Pac-Man's current cell.

If Blinky is frightened, the rule is replaced by: move to maximize BFS distance from Pac-Man on the ghost adjacency graph (paper §IV-F Case 2 from the other paper — borrow it).

### 3.5 The MCTS loop (per ghost)

For each MCTS ghost G, run the standard loop until the compute budget is spent:

```
loop:
    1. SELECTION: descend from root by UCB1 until reaching a leaf node
    2. EXPANSION: add the leaf's children if not yet expanded; pick one child
    3. SIMULATION: roll out from the child's state (see §3.6)
    4. BACKPROP: add the rollout reward to X_i and increment T_i along the visited path
```

When G actually reaches the cross-point that is its current root (in real game ticks), **discard its tree and start a new search rooted at the new next-cross-point.** Also, on a global reverse event (§3.8), all three trees are discarded and rebuilt.

The move played by G is, at each cross-point, the child of the root with the **highest mean reward `X_i / T_i`** (tiebreak: highest `T_i`). Pick once per arrival at a cross-point and commit until the next cross-point.

### 3.6 Simulation rules

Inside one rollout (the "playout phase"):

**The MCTS ghost G of interest** moves according to its own tree along the path being evaluated, then random at cross-points once it exits the tree.

**The two other MCTS ghosts** move according to *their current, frozen, in-use trees* (from the last completed search) — descend by UCB1 / best-child, random at the leaves. This is the trick that gives emergent coordination. If a ghost has no tree yet (early game), it moves randomly at cross-points.

**Blinky** moves randomly at cross-points during simulation. **This is a deliberate inversion of its real-game rule** — the paper claims this makes the MCTS ghosts plan more aggressively, since they can't assume Blinky's pressure for free. Keep it. Document the choice prominently in the design doc; it will surprise readers.

**Pac-Man** moves according to the KNN predictor (§4) with probability `p_predict`, otherwise uniformly random over legal directions at cross-points. `p_predict` decays linearly from `p_predict_start` (default 0.9) to `p_predict_end` (default 0.1) over the rollout's `T_sim` ticks. The paper's reasoning: the predictor is approximate, so late-rollout moves should hedge with random exploration.

**Stopping conditions** (already match your game — no adaptation needed):
1. Pac-Man eaten by a non-frightened ghost → big reward signal (see §3.7).
2. All pellets and energizers eaten → bad outcome for ghosts.
3. `T_sim` ticks have elapsed → score what we have.

### 3.7 Reward signal

The paper specifies three components: reverse of Pac-Man's score, reverse of time, dispersal penalty. Concretize as:

```
R = w_capture · 1[pacman_eaten]
  + w_deny    · (1 − pellets_eaten_during_sim / pellets_at_root)
  + w_time    · (1 − sim_ticks / T_sim)
  + w_disperse · dispersal_score
  − w_eaten   · (frightened_ghosts_eaten / 4)
```

where `dispersal_score = mean pairwise BFS distance among ghosts at sim end, normalized by graph diameter`.

Starting weights: `w_capture = 1.0`, `w_deny = 0.3`, `w_time = 0.1`, `w_disperse = 0.1`, `w_eaten = 0.5`. Tunable in `IceGuctConfig`.

Apply this reward to **every node along the path from root to the expanded leaf** — i.e. accumulated reward backprop, paper §III-A step 4.

### 3.8 Tree rebuild triggers

Each ghost's tree is rebuilt when:
1. The ghost arrives at the cross-point that is its current root (advance the search frontier).
2. A **global reverse event** fires — in this game, the tick an energizer is eaten by Pac-Man (subject to user's answer to Q1 in §2). Detect this by watching for any `ghost_frightened[i]` transition from 0 to nonzero. On this event, **all three MCTS ghosts discard their trees** and start over from their new next-cross-points.

### 3.9 Compute budget allocation (§ depends on user's answer to Q4)

Two sensible options, propose both in the design doc and pick one:

- **Sequential, equal split.** Each MCTS ghost gets `budget / 3` per tick. Simple, deterministic. Likely the right default.
- **Sequential, "needs-based" split.** A ghost whose tree was just rebuilt (e.g., after a global reverse) gets a larger share that tick because it has zero stats; otherwise equal split.

Parallel execution across ghosts is also possible if the simulator is thread-safe — leave that as a future optimization, not a v1 requirement.

---

## 4. KNN Pac-Man movement predictor

### 4.1 Feature vector (the "input space")

Per the paper §IV-A, for the current game state:

| # | Feature | Notes |
|---|---|---|
| 1 | distance to nearest ghost | BFS distance on Pac-Man's walkable graph |
| 2 | distance to 2nd-nearest ghost | same |
| 3 | distance to nearest energizer | same |
| 4 | distance to nearest pellet | same |
| 5 | distance to nearest cross-point | same |
| 6 | nearest ghost edible? | 0 or 1, derived from `ghost_frightened[i] > 0` |
| 7 | 2nd-nearest ghost edible? | 0 or 1 |
| 8 | nearest-energizer's-nearest-ghost edible? | 0 or 1 |
| 9 | distance from Pac-Man to (nearest energizer's nearest ghost) | BFS distance |
| 10 | distance from nearest energizer to its nearest ghost | BFS distance |

Normalize all distances by graph diameter to put them on the same scale. Use Pac-Man's walkable graph (excluding ghost house) for these BFS distances — that's the graph Pac-Man can actually travel.

### 4.2 The "move space" and prediction

Same as paper §IV-B/C: for each training example we store a *move vector* `γ` that describes the meaning of Pac-Man's actual move in that state — concretely, the vector difference in normalized feature values from the pre-move state to the post-move state (or a fixed-length subset of features that capture "got closer to X, further from Y"). Then:

```
γ̂_p = Σ_{i=1..k} (γ_i / d(p,i))  /  Σ_{i=1..k} (1 / d(p,i))
```

with `k = 3` (paper's choice). Then enumerate `Γ_p = {γ_{p,dir} | dir ∈ legal moves at p}` — i.e. for each legal direction, the move vector you *would* get by attempting that direction from `p` — and pick the direction whose `γ_{p,dir}` is closest to `γ̂_p`.

### 4.3 Implementation notes

- Use scikit-learn's `BallTree` or `KDTree` for the nearest-neighbor index. With ~1000s of trajectory examples and a 10-dim feature space, lookup is sub-millisecond.
- Cache the index. Rebuilding it per call is unacceptable in the inner loop of MCTS rollouts.
- If the training set is empty (cold start), the predictor returns "random legal direction" and the agent logs a warning at construction time.
- Provide a `KNNPredictor.score(test_set)` method that reports prediction accuracy — useful for tuning.

---

## 5. KNN data-collection script

`scripts/collect_trajectories.py`

**Purpose.** Generate a training dataset of Pac-Man (state, move) pairs for the KNN predictor.

**Usage.**
```
python scripts/collect_trajectories.py \
    --maze-dir mazes/ \
    --pacman heuristic \          # or "mcts" or any registered Pac-Man agent
    --ghosts random \             # or "heuristic"
    --n-games 100 \
    --seed 42 \
    --out data/knn_train/
```

**Behaviour.**
1. For each game, pick a maze (round-robin over the directory) and run a full game using the existing game runner.
2. Save the resulting session to `data/knn_train/session_<id>.pkl` in whatever session format the existing codebase uses.
3. Extract `(feature_vector, move_vector)` pairs at every Pac-Man decision tick (cross-point or direction change) and append to `data/knn_train/dataset.npz` (two arrays: `X` shape `(N, 10)`, `Y` shape `(N, move_vec_dim)`).
4. Print summary: games played, outcomes breakdown, total pairs collected.

**Default Pac-Man for bootstrap.** Use the existing heuristic Pac-Man (the rollout policy already in the codebase). It's deterministic enough to be reproducible and gives the KNN a sensible behavior to imitate. Note: ideally we'd train on the JEPA Pac-Man trajectories, but the ghost agent must work *before* JEPA exists, so we bootstrap.

**Re-training.** The KNN predictor reloads `dataset.npz` at construction time. To upgrade the predictor later (e.g., after JEPA is trained), just re-run this script with `--pacman jepa` and the ghost agent picks it up next game.

---

## 6. Integration with the existing codebase

### 6.1 New agent class

```python
# src/agents/iceguct_ghosts.py
class IceGuctGhosts:
    name = "iceguct"
    def __init__(self, config: IceGuctConfig | None = None,
                 knn_dataset_path: str | Path = "data/knn_train/dataset.npz",
                 rng: np.random.Generator | None = None): ...
    def reset(self, maze: Maze, role: Literal["ghosts"]) -> None: ...
    def act(self, observation: Observation) -> tuple[Direction, Direction, Direction, Direction]: ...
```

Implements the same `Agent` protocol as the rest. Returns a tuple of 4 directions per tick (one per ghost in the simulator's ghost order).

### 6.2 Agent registry

Register under the name `"iceguct"` in whatever registry the codebase uses (`AGENT_REGISTRY` or similar). The benchmark harness and `scripts/play.py` should then accept `--ghosts iceguct` with no other changes.

### 6.3 Configuration

```python
@dataclass
class IceGuctConfig:
    C: float = 1.41                 # UCB1 exploration
    epsilon: float = 1e-6           # UCB1 denominator term
    T_sim: int = 80                 # max rollout ticks
    k_neighbors: int = 3            # KNN
    p_predict_start: float = 0.9    # Pac-Man predictor trust at sim start
    p_predict_end: float = 0.1      # at sim end
    # Reward weights
    w_capture: float = 1.0
    w_deny: float = 0.3
    w_time: float = 0.1
    w_disperse: float = 0.1
    w_eaten: float = 0.5
    # Budget
    budget_ms_per_tick: int = 40    # total for all 3 MCTS ghosts
    budget_split: Literal["equal", "needs_based"] = "equal"
```

### 6.4 Logging

At `INFO`: per-tick total iterations across the 3 trees, tree sizes, global-reverse events, KNN dataset size at startup.

At `DEBUG`: per-ghost iteration count, chosen direction, top child stats `(direction, X_i/T_i, T_i)`.

### 6.5 Files to delete

The previous joint-action ghost agent. Identify the file(s) by reading the registry and the existing agent modules; remove them from the registry and delete the source. Confirm with the user before deletion if anything ambiguous.

---

## 7. Tests

Add under `tests/`:

1. **`test_ghost_graph_includes_ghost_house.py`** — assert the ghost adjacency graph contains every `ghost_walls` cell, and assert it differs from the Pac-Man graph *exactly* on the ghost-house subgraph.

2. **`test_blinky_chase.py`** — given a synthetic state with Pac-Man heading east toward a known junction, assert Blinky's chosen move reduces BFS distance to that junction.

3. **`test_blinky_flee_when_frightened.py`** — same setup but Blinky frightened; assert the chosen move *increases* BFS distance to Pac-Man.

4. **`test_tree_rebuild_on_global_reverse.py`** — run a game until Pac-Man eats an energizer; assert all three MCTS trees were discarded and rebuilt within one tick after the event.

5. **`test_knn_cold_start.py`** — construct the agent with an empty/missing dataset; assert it warns, falls back to random-direction prediction, and does not crash.

6. **`test_knn_prediction_shape.py`** — load a small synthetic dataset and assert `predict(state)` returns a legal direction in O(ms) time.

7. **`test_iceguct_vs_random_pacman.py`** (sanity / smoke) — run 10 short games of the new ghost agent vs a random-direction Pac-Man on the example maze. Assert the ghost agent wins (Pac-Man eaten) in ≥ 8 of 10. If this fails, something is fundamentally broken.

8. **`test_iceguct_compute_budget.py`** — assert `act()` returns within `1.2 × budget_ms_per_tick` for 100 consecutive ticks (some slop for OS scheduling).

9. **`test_data_collection_script.py`** — invoke `collect_trajectories.py` with `--n-games 2` against a temp output dir; assert two sessions written and `dataset.npz` is non-empty and well-shaped.

All tests use a fixed RNG seed for determinism.

---

## 8. Documentation deliverables

### 8.1 `docs/USAGE.md` — add a new section

> ### Ghost agent: `iceguct`
> How to construct it, where the KNN dataset lives, how to (re)collect data, configuration knobs, troubleshooting (empty dataset → warning, slow act() → reduce T_sim or budget, etc.).

### 8.2 `docs/DESIGN.md` — add or replace the ghost-team section

Structure mirrors the existing DESIGN.md sections. Cover:

1. **Why the previous joint-action MCTS was replaced.** One paragraph — branching factor (4⁴ vs 4), no competition pedigree, no opponent model.
2. **ICE gUCT overview.** Per-ghost trees + Blinky rule + KNN predictor.
3. **Tree structure and tree-rebuild triggers.** With a diagram showing root = next-cross-point and the rebuild on global reverse.
4. **The emergent-coordination trick.** "Other ghosts move per their frozen trees during my simulation" — explain why this produces team behavior without a joint action space.
5. **The in-simulation-Blinky-random trick.** Explicitly call out that this is non-obvious. The user will get questions about it from readers and during the presentation; arm them with the answer: it forces the MCTS ghosts to plan as if Blinky won't help, which produces more aggressive plans.
6. **Reward components and weights.** Show the formula from §3.7 and justify each weight's order of magnitude.
7. **KNN predictor — input space, move space, prediction algorithm.** Include the two-space diagram from the paper (or recreate it).
8. **Adaptations vs. the paper.** Tabulate: kept (per-ghost trees, KNN, UCB1, Blinky rule, in-sim Blinky random), changed (global-reverse trigger, terminal conditions match the no-levels game), new (explicit dispersal score, decayed predictor trust).
9. **Cold-start strategy.** How the KNN is bootstrapped with heuristic-Pac-Man trajectories before JEPA exists.
10. **Limitations.** Single accumulated reward (no max/avg split like Pepels Pac-Man), no explicit endgame handling for ghosts, KNN is a weak learner — it only models the Pac-Man it was trained on.

The user will lift sections directly into their report and presentation, so write them with that in mind.

---

## 9. Acceptance criteria

You're done when all of these hold:

- [ ] `python scripts/play.py --pacman <existing> --ghosts iceguct --maze <any>` runs without crashing.
- [ ] All existing tests still pass (i.e., you haven't broken the Pac-Man agent, simulator, or viewer).
- [ ] All 9 new tests in §7 pass.
- [ ] The old joint-action ghost agent is fully removed (file deleted, registry entry deleted, no dead imports).
- [ ] `data/knn_train/dataset.npz` can be (re-)generated by `scripts/collect_trajectories.py` from any maze directory.
- [ ] `docs/USAGE.md` and `docs/DESIGN.md` have the new sections in §8.
- [ ] On the example maze, the ICE gUCT ghost team paired with the existing MCTS Pac-Man produces game lengths and outcomes that are *plausible* (Pac-Man doesn't trivially win every game, ghosts don't trivially win every game). No strict win-rate target — that's for the benchmark.

---

## 10. Process notes

- **Re-confirm Q1–Q4 from §2 before writing any algorithm code.** All four answers shape the simulation and the rebuild triggers.
- Keep PR-sized commits: (1) graph + Blinky rule + tests, (2) KNN predictor + data collection + tests, (3) MCTS loop + reward + tests, (4) integration + remove old agent + docs.
- The paper is short (4 pages) and under-specifies several constants (UCB `C`, the max-revisits cap, reward weights). Use the defaults in §3.7 and `IceGuctConfig`; document any that you change.
- If something in the paper is genuinely ambiguous (e.g., what exactly "reverse of time" means as a number), pick the most defensible interpretation, implement it, and note the choice in `DESIGN.md` §8 under "Adaptations vs. the paper." Don't silently invent.
