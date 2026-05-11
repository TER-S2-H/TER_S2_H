import argparse
import multiprocessing
import json
import math
import random
import time
from dataclasses import dataclass
from itertools import repeat
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    from pacman_map import PacmanMap
    from maze_complexity import compute_complexity_score
except ImportError:
    from .pacman_map import PacmanMap
    from .maze_complexity import compute_complexity_score


FEATURE_KEYS = [
    "node_count",
    "edge_count",
    "score",
    "steps",
    "longest_trap",
    "avg_trap_len",
    "num_traps",
    "num_tunnels",
]


DEFAULT_WEIGHTS: Dict[str, float] = {
    "node_count": 0.5,
    "edge_count": 0.5,
    "score": 3.0,
    "steps": 0.5,
    "longest_trap": 1.0,
    "avg_trap_len": 1.5,
    "num_traps": 2.0,
    "num_tunnels": 2.0,
}


_RANGES: Dict[str, float] = {
    "node_count": 50,
    "edge_count": 50,
    "score": 5.0,
    "steps": 200,
    "longest_trap": 10,
    "avg_trap_len": 5.0,
    "num_traps": 10,
    "num_tunnels": 1,
}


SEED_MIN = 0
SEED_MAX = 2**32 - 1
SEED_MOD = 2**32


@dataclass
class SearchResult:
    seed: int
    features: Dict[str, float]
    distance: float
    evaluations: int
    algorithm: str


def generate_maze(seed: int) -> Dict[str, Any]:
    maze = PacmanMap(seed=seed)
    maze.generate()
    graph = maze.export_to_graph_json()
    metadata = graph.setdefault("metadata", {})
    metadata["complexity_score"] = compute_complexity_score(
        graph,
        start=(13, 23),
        wrap_tunnels=True,
    )
    return graph


def extract_features(maze_json: Dict[str, Any]) -> Dict[str, float]:
    metadata = maze_json["metadata"]
    complexity = metadata["complexity_score"]
    return {
        "node_count": metadata["node_count"],
        "edge_count": metadata["edge_count"],
        "score": complexity["score"],
        "steps": complexity["steps"],
        "longest_trap": complexity["longest_trap"],
        "avg_trap_len": complexity["avg_trap_len"],
        "num_traps": complexity["num_traps"],
        "num_tunnels": complexity["num_tunnels"],
    }


def validate_target(target: Dict[str, float]) -> List[str]:
    warnings: List[str] = []

    unknown = [key for key in target.keys() if key not in FEATURE_KEYS]
    if unknown:
        warnings.append(f"Unknown target keys ignored by distance: {unknown}")

    if "num_tunnels" in target and target["num_tunnels"] not in (1, 2):
        warnings.append("Target warning: num_tunnels should be 1 or 2.")

    if "score" in target and not (2.0 <= float(target["score"]) <= 12.0):
        warnings.append("Target warning: score is outside plausible range [2.0, 12.0].")

    return warnings


def feature_distance(
    actual: Dict[str, float],
    target: Dict[str, float],
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
) -> float:
    total = 0.0
    for key, t_val in target.items():
        if key not in actual:
            continue
        w = weights.get(key, 1.0)
        norm = _RANGES.get(key, 1.0)
        total += w * ((actual[key] - t_val) / norm) ** 2
    return math.sqrt(total)


def _clamp_seed(seed: int) -> int:
    return max(SEED_MIN, min(SEED_MAX, seed))


def _print_target_delta(actual: Dict[str, float], target: Dict[str, float]) -> None:
    if not target:
        print("No target features were specified.")
        return

    print("Target deltas:")
    for key in FEATURE_KEYS:
        if key not in target or key not in actual:
            continue
        t_val = target[key]
        a_val = actual[key]
        diff = a_val - t_val
        print(f"  {key}: target={t_val}, actual={a_val}, diff={diff:+.4f}")


def _print_result(result: SearchResult, target: Dict[str, float]) -> None:
    print(f"\n[{result.algorithm}] best seed: {result.seed}")
    print(f"[{result.algorithm}] best distance: {result.distance:.6f}")
    print(f"[{result.algorithm}] evaluations: {result.evaluations}")
    print(f"[{result.algorithm}] features: {result.features}")
    _print_target_delta(result.features, target)
    if result.distance < 0.15:
        print("Target reached within tolerance.")


def _evaluate_seed_for_target(
    seed: int,
    target: Dict[str, float],
    weights: Dict[str, float],
) -> Tuple[int, Dict[str, float], float]:
    features = extract_features(generate_maze(seed))
    distance = feature_distance(features, target, weights)
    return seed, features, distance


def search_random(
    target: Dict[str, float],
    n_seeds: int = 100_000,
    seed_range: Tuple[int, int] = (0, 2**32),
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
    workers: int = 1,
) -> SearchResult:
    rng = random.Random()
    low, high = seed_range
    low = max(SEED_MIN, low)
    high = min(SEED_MOD, high)
    if low >= high:
        raise ValueError("Invalid seed_range; expected low < high within [0, 2**32].")

    best_seed: Optional[int] = None
    best_features: Optional[Dict[str, float]] = None
    best_distance = float("inf")

    seeds = [rng.randrange(low, high) for _ in range(n_seeds)]

    if workers <= 1:
        for index, seed in enumerate(seeds, start=1):
            features = extract_features(generate_maze(seed))
            distance = feature_distance(features, target, weights)

            if distance < best_distance:
                best_distance = distance
                best_seed = seed
                best_features = features

            if index % 10_000 == 0:
                print(f"[random] evaluated {index}/{n_seeds}, best_distance={best_distance:.6f}")
    else:
        processes = min(max(1, workers), multiprocessing.cpu_count())
        chunk = max(1, n_seeds // (processes * 8))
        with multiprocessing.Pool(processes=processes) as pool:
            iterator = pool.starmap(
                _evaluate_seed_for_target,
                zip(seeds, repeat(target), repeat(weights)),
                chunksize=chunk,
            )
            for index, (seed, features, distance) in enumerate(iterator, start=1):
                if distance < best_distance:
                    best_distance = distance
                    best_seed = seed
                    best_features = features

                if index % 10_000 == 0:
                    print(f"[random] evaluated {index}/{n_seeds}, best_distance={best_distance:.6f}")

    if best_seed is None or best_features is None:
        raise RuntimeError("Random search failed to evaluate any seed.")

    result = SearchResult(
        seed=best_seed,
        features=best_features,
        distance=best_distance,
        evaluations=n_seeds,
        algorithm="random",
    )
    _print_result(result, target)
    return result


def search_hill_climb(
    target: Dict[str, float],
    start_seed: Optional[int] = None,
    steps: int = 500,
    restarts: int = 5,
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
) -> SearchResult:
    rng = random.Random()
    deltas = [1, 10, 100, 1_000, 10_000, 100_000]
    total_evaluations = 0

    best_global_seed: Optional[int] = None
    best_global_features: Optional[Dict[str, float]] = None
    best_global_distance = float("inf")

    for restart in range(restarts):
        if start_seed is None:
            current_seed = rng.randrange(SEED_MIN, SEED_MOD)
        else:
            current_seed = _clamp_seed(start_seed)

        current_features = extract_features(generate_maze(current_seed))
        current_distance = feature_distance(current_features, target, weights)
        total_evaluations += 1

        for _ in range(steps):
            delta = rng.choice(deltas)
            direction = rng.choice([-1, 1])
            candidate_seed = _clamp_seed(current_seed + direction * delta)

            candidate_features = extract_features(generate_maze(candidate_seed))
            candidate_distance = feature_distance(candidate_features, target, weights)
            total_evaluations += 1

            if candidate_distance < current_distance:
                current_seed = candidate_seed
                current_features = candidate_features
                current_distance = candidate_distance

        print(
            f"[hill] restart {restart + 1}/{restarts}, "
            f"best_distance_this_restart={current_distance:.6f}"
        )

        if current_distance < best_global_distance:
            best_global_seed = current_seed
            best_global_features = current_features
            best_global_distance = current_distance

    if best_global_seed is None or best_global_features is None:
        raise RuntimeError("Hill climbing failed to evaluate any seed.")

    result = SearchResult(
        seed=best_global_seed,
        features=best_global_features,
        distance=best_global_distance,
        evaluations=total_evaluations,
        algorithm="hill_climb",
    )
    _print_result(result, target)
    return result


def search_cma(
    target: Dict[str, float],
    budget: int = 2_000,
    popsize: int = 16,
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
) -> SearchResult:
    try:
        import cma
    except ImportError as error:
        raise ImportError("search_cma requires the 'cma' package. Install with: pip install cma") from error

    evaluations = 0
    best_seed: Optional[int] = None
    best_features: Optional[Dict[str, float]] = None
    best_distance = float("inf")

    x0 = [float(random.randrange(SEED_MIN, SEED_MOD))]
    es = cma.CMAEvolutionStrategy(x0, 500_000, {"popsize": popsize, "verbose": -9})

    while evaluations < budget and not es.stop():
        remaining_budget = budget - evaluations
        if remaining_budget < es.sp.mu:
            break

        population = es.ask()
        values: List[float] = []
        evaluated_population: List[Sequence[float]] = []

        for candidate in population:
            if len(evaluated_population) >= remaining_budget:
                break

            seed = abs(int(candidate[0])) % SEED_MOD
            features = extract_features(generate_maze(seed))
            distance = feature_distance(features, target, weights)

            evaluations += 1
            values.append(distance)
            evaluated_population.append(candidate)

            if distance < best_distance:
                best_distance = distance
                best_seed = seed
                best_features = features

        if not values:
            break

        es.tell(evaluated_population, values)

    if best_seed is None or best_features is None:
        raise RuntimeError("CMA-ES failed to evaluate any seed.")

    result = SearchResult(
        seed=best_seed,
        features=best_features,
        distance=best_distance,
        evaluations=evaluations,
        algorithm="cma_es",
    )
    _print_result(result, target)
    return result


def benchmark(
    target: Dict[str, float],
    runs: int = 3,
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
) -> List[SearchResult]:
    algorithm_runs: List[Tuple[str, Callable[[], SearchResult]]] = [
        ("random", lambda: search_random(target=target, weights=weights)),
        ("hill_climb", lambda: search_hill_climb(target=target, weights=weights)),
        ("cma_es", lambda: search_cma(target=target, weights=weights)),
    ]

    summaries: List[Tuple[str, SearchResult, float, int]] = []

    print("\nBenchmark: algorithm | best distance | evaluations | time (s)")
    for name, run_function in algorithm_runs:
        best: Optional[SearchResult] = None
        total_time = 0.0
        total_evaluations = 0

        for run_index in range(1, runs + 1):
            start = time.perf_counter()
            result = run_function()
            elapsed = time.perf_counter() - start

            total_time += elapsed
            total_evaluations += result.evaluations
            if best is None or result.distance < best.distance:
                best = result

            print(
                f"[benchmark] {name} run {run_index}/{runs}: "
                f"distance={result.distance:.6f}, evals={result.evaluations}, time={elapsed:.2f}s"
            )

        if best is None:
            raise RuntimeError(f"No benchmark result for algorithm: {name}")

        summaries.append((name, best, total_time, total_evaluations))

    print("\nalgorithm | best distance | evaluations | time (s)")
    for name, best, total_time, total_evaluations in summaries:
        print(f"{name:9s} | {best.distance:13.6f} | {total_evaluations:11d} | {total_time:8.2f}")

    results = [best for _, best, _, _ in summaries]
    results.sort(key=lambda result: result.distance)
    return results


def _parse_json_dict(value: str) -> Dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def _parse_weights(weights_json: Optional[str]) -> Dict[str, float]:
    if weights_json is None:
        return dict(DEFAULT_WEIGHTS)
    parsed = _parse_json_dict(weights_json)
    return {str(key): float(value) for key, value in parsed.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Pacman maze seeds by target features.")
    parser.add_argument(
        "--target",
        type=str,
        default='{"score": 7.0, "num_traps": 25, "num_tunnels": 2}',
        help="JSON dict target, e.g. '{\"score\": 7.0, \"num_traps\": 25}'.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["random", "hill", "cma", "all"],
        default="all",
        help="Search algorithm to run.",
    )
    parser.add_argument("--weights", type=str, default=None, help="Optional JSON dict of feature weights.")

    parser.add_argument("--n-seeds", type=int, default=100_000)
    parser.add_argument("--seed-min", type=int, default=0)
    parser.add_argument("--seed-max", type=int, default=2**32)
    parser.add_argument("--workers", type=int, default=1)

    parser.add_argument("--start-seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--restarts", type=int, default=5)

    parser.add_argument("--budget", type=int, default=2_000)
    parser.add_argument("--popsize", type=int, default=16)
    parser.add_argument("--runs", type=int, default=3)

    args = parser.parse_args()

    target = _parse_json_dict(args.target)
    weights = _parse_weights(args.weights)

    specified = [key for key in FEATURE_KEYS if key in target]
    print(f"Specified target features: {specified}")
    for warning in validate_target(target):
        print(warning)

    if args.algorithm == "random":
        search_random(
            target=target,
            n_seeds=args.n_seeds,
            seed_range=(args.seed_min, args.seed_max),
            weights=weights,
            workers=args.workers,
        )
    elif args.algorithm == "hill":
        search_hill_climb(
            target=target,
            start_seed=args.start_seed,
            steps=args.steps,
            restarts=args.restarts,
            weights=weights,
        )
    elif args.algorithm == "cma":
        search_cma(
            target=target,
            budget=args.budget,
            popsize=args.popsize,
            weights=weights,
        )
    else:
        results = benchmark(target=target, runs=args.runs, weights=weights)
        print("\nSorted benchmark winners:")
        for result in results:
            print(f"  {result.algorithm}: seed={result.seed}, distance={result.distance:.6f}")


if __name__ == "__main__":
    main()
