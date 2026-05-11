"""
Pipeline : entraînement supervisé des cost modules → fine-tuning RL.

Étapes :
  1. Entraînement supervisé (Optuna + réentraînement final)
     → optuna_results/cost_pacman.pt + cost_ghost.pt
  2. Fine-tuning par renforcement
     → optuna_results/cost_pacman_rl.pt + cost_ghost_rl.pt

Usage :
    python -m src.jepacman.pipeline \
        --obs-dir      jepa_numpy \
        --meta-dir     jepa_meta \
        --jepa-weights optuna_results/best_model.pt \
        --jepa-config  optuna_results/best_config_full.json \
        --maze-json    path/to/maze.json
"""

import argparse
import os
import sys


OUTPUT_DIR = "optuna_results"


def step_supervised(obs_dir, meta_dir, jepa_weights, jepa_config,
                    n_trials, storage):
    print("\n" + "=" * 60)
    print("ÉTAPE 1 — Entraînement supervisé des modules de coût")
    print("=" * 60)

    from src.jepacman.cost_train_supervised import run
    run(
        obs_dir      = obs_dir,
        meta_dir     = meta_dir,
        jepa_weights = jepa_weights,
        jepa_config  = jepa_config,
        n_trials     = n_trials,
        storage      = storage,
    )

    for fname in ("cost_pacman.pt", "cost_ghost.pt"):
        path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Entraînement supervisé terminé mais {path} introuvable."
            )
    print(f"\n✓ Poids supervisés disponibles dans {OUTPUT_DIR}/")


def step_rl(jepa_weights, jepa_config, maze_dir,
            n_episodes, rl_lr, save_every, render, render_every, render_fps):
    print("\n" + "=" * 60)
    print("ÉTAPE 2 — Fine-tuning par renforcement")
    print("=" * 60)

    from src.jepacman.rl_loop import run_rl
    run_rl(
        maze_dir = maze_dir,
        jepa_weights = jepa_weights,
        jepa_config  = jepa_config,
        n_episodes   = n_episodes,
        lr           = rl_lr,
        save_every   = save_every,
        render=render, 
        render_every=render_every,
        render_fps=render_fps
    )
    print("\n✓ Fine-tuning RL terminé")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline supervisé → RL pour les modules de coût"
    )

    # Données
    parser.add_argument("--obs-dir",      required=None)
    parser.add_argument("--meta-dir",     required=None)
    parser.add_argument("--jepa-weights", required=True)
    parser.add_argument("--jepa-config",  required=True)
    parser.add_argument("--maze-dir", default="generated_mazes")

    # Supervisé
    parser.add_argument("--n-trials",  type=int, default=7)
    parser.add_argument("--storage",   type=str, default=None,
                        help="ex: sqlite:///optuna_cost.db")

    # RL
    parser.add_argument("--n-episodes",  type=int,   default=200)
    parser.add_argument("--rl-lr",       type=float, default=1e-4)
    parser.add_argument("--save-every",  type=int,   default=50)
    parser.add_argument("--render",        action="store_true")
    parser.add_argument("--render-every",  type=int,   default=10)
    parser.add_argument("--render-fps",    type=int,   default=12)

    # Reprises partielles
    parser.add_argument("--skip-supervised", action="store_true",
                        help="Sauter l'étape supervisée (utilise les poids existants)")
    parser.add_argument("--skip-rl",         action="store_true",
                        help="Sauter l'étape RL")

    args = parser.parse_args()

    try:
        if not args.skip_supervised:
            step_supervised(
                obs_dir      = args.obs_dir,
                meta_dir     = args.meta_dir,
                jepa_weights = args.jepa_weights,
                jepa_config  = args.jepa_config,
                n_trials     = args.n_trials,
                storage      = args.storage,
            )
        else:
            print("\n⚠ Étape supervisée ignorée (--skip-supervised)")
            for fname in ("cost_pacman.pt", "cost_ghost.pt"):
                path = os.path.join(OUTPUT_DIR, fname)
                if not os.path.exists(path):
                    print(f"  ⚠ {path} introuvable — le RL démarrera "
                          "avec des modules de coût non entraînés")

        if not args.skip_rl:
            step_rl(
                jepa_weights = args.jepa_weights,
                jepa_config  = args.jepa_config,
                maze_dir     = args.maze_dir,
                n_episodes   = args.n_episodes,
                rl_lr        = args.rl_lr,
                save_every   = args.save_every,
                render       = args.render,
                render_every = args.render_every,
                render_fps   = args.render_fps,
            )
        else:
            print("\n⚠ Étape RL ignorée (--skip-rl)")

    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrompu.")
        print("  → Relancez avec --skip-supervised pour reprendre au RL.")
        sys.exit(1)

    except Exception as e:
        print(f"\n✗ Erreur : {e}")
        raise

    print("\n" + "=" * 60)
    print("PIPELINE TERMINÉ")
    print(f"  Poids supervisés : {OUTPUT_DIR}/cost_pacman.pt + cost_ghost.pt")
    print(f"  Poids RL         : {OUTPUT_DIR}/cost_pacman_rl.pt + cost_ghost_rl.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()