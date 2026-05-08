#!/bin/bash
# pipeline.sh — à placer à la racine du projet (au-dessus de src/)
# Usage: bash pipeline.sh --n-mazes 10 --n-sessions 1000

set -e

export PYTHONPATH="$PWD/src:$PYTHONPATH"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
# ── Paramètres ────────────────────────────────────────────────────────────────
N_MAZES=10
N_SESSIONS=1000
MAX_STEPS=3000
SEEDS=""
STORAGE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --n-mazes)    N_MAZES="$2";    shift ;;
        --n-sessions) N_SESSIONS="$2"; shift ;;
        --max-steps)  MAX_STEPS="$2";  shift ;;
        --seeds)      SEEDS="$2";      shift ;;
        --storage)    STORAGE="$2";    shift ;;
    esac
    shift
done

# ── Chemins ───────────────────────────────────────────────────────────────────
SRC="src"
MAZE_DIR="generated_mazes"
SESSION_DIR="sessions"
JEPA_DIR="jepa_sessions"

mkdir -p $MAZE_DIR $SESSION_DIR $JEPA_DIR

# ── Étape 1 — Générer les labyrinthes ─────────────────────────────────────────
echo "=== Étape 1 : Génération des labyrinthes ==="

if [ -z "$SEEDS" ]; then
    for i in $(seq 1 $N_MAZES); do
        python -m src.generate -o $MAZE_DIR/maze_${RANDOM} --verbose
    done
else
    for seed in $SEEDS; do
        python -m src.generate -o $MAZE_DIR/maze_${seed} --seed $seed --verbose
    done
fi

echo "Labyrinthes générés dans $MAZE_DIR/"

# ── Étape 2 — Simuler les parties ─────────────────────────────────────────────
echo ""
echo "=== Étape 2 : Simulation des parties ==="

for maze_dir in $MAZE_DIR/maze_*/; do
    maze_json="$maze_dir/maze.json"
    seed=$(basename $maze_dir | sed 's/maze_//')

    if [ ! -f "$maze_json" ]; then
        echo "Pas de maze.json dans $maze_dir — ignoré"
        continue
    fi

    echo "Simulation seed=$seed ($N_SESSIONS parties)..."
    python $SRC/pacman_simulator.py \
        $maze_json \
        -o $SESSION_DIR \
        -n $N_SESSIONS \
        --max-steps $MAX_STEPS \
        --seed $seed
done

echo "Sessions sauvegardées dans $SESSION_DIR/"

# ── Étape 3 — Convertir au format JEPA ────────────────────────────────────────
echo ""
echo "=== Étape 3 : Conversion au format JEPA ==="

python $SRC/jepacman/convert_sess_to_jepa.py \
    $SESSION_DIR \
    -o $JEPA_DIR \
    --recurse

echo "Sessions converties dans $JEPA_DIR/"

# ── Étape 4 — Entraînement ────────────────────────────────────────────────────
echo ""
echo "=== Étape 4 : Entraînement ==="

STORAGE_ARG=""
if [ ! -z "$STORAGE" ]; then
    STORAGE_ARG="--storage $STORAGE"
fi

python -m src.jepacman.train $JEPA_DIR $STORAGE_ARG

echo ""
echo "=== Pipeline terminée ==="
echo "Résultats dans optuna_results/"