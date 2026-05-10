"""
Entraînement supervisé des modules de coût avec optimisation Optuna.

Usage :
    python -m src.jepacman.cost_train_supervised \\
        --obs-dir   jepa_numpy \\
        --meta-dir  jepa_meta \\
        --jepa-weights optuna_results/best_model.pt \\
        --jepa-config  optuna_results/best_config_full.json \\
        --n-trials  20 \\
        --storage   sqlite:///optuna_cost.db
"""

import os
import gc
import json
import argparse
import dataclasses

import torch
import pytorch_lightning as pl
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)
from torch.utils.data import DataLoader, random_split

from .config import Config
from .models.jepa import HierarchicalJEPAcman
from .models.cost_module import CostModule
from .data.cost_dataset import CostDataset
from .lightning.cost_module_pl import CostModuleModule

# ── Constantes ────────────────────────────────────────────────────────────────

N_TRIALS       = 5
MAX_EPOCHS_TRIALS = 15
MAX_EPOCHS     = 30
EARLY_STOPPING = 7
VAL_SPLIT      = 0.1
OUTPUT_DIR     = "optuna_results_cost"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jepa(jepa_weights: str, jepa_config: str) -> tuple:
    """Charge le modèle JEPA et sa config depuis le disque."""
    with open(jepa_config) as f:
        cfg_raw = json.load(f)
    valid  = {f.name for f in dataclasses.fields(Config)}
    config = Config(**{k: v for k, v in cfg_raw.items() if k in valid})

    jepa = HierarchicalJEPAcman(
        case_dim=config.case_dim, n_actions=config.n_actions,
        d_model=config.d_model, seq_len=config.seq_len,
        n_heads_state=config.n_heads_state, n_layers_state=config.n_layers_state,
        n_heads_seq=config.n_heads_seq, n_layers_seq=config.n_layers_seq,
        n_heads_pred_1=config.n_heads_pred_1, n_layers_pred_1=config.n_layers_pred_1,
        n_heads_pred_2=config.n_heads_pred_2, n_layers_pred_2=config.n_layers_pred_2,
        n_heads_pred_3=config.n_heads_pred_3, n_layers_pred_3=config.n_layers_pred_3,
        ema_momentum=config.ema_momentum, k=config.k, j=config.j,
    )
    jepa.load_state_dict(torch.load(jepa_weights, map_location="cpu"))
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad = False

    return jepa, config


def make_loaders(dataset: CostDataset, batch_size: int):
    n_val   = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    cuda = torch.cuda.is_available()
    kw = dict(
        batch_size=batch_size,
        num_workers=4,
        pin_memory=cuda,
        persistent_workers=True,
    )
    return DataLoader(train_ds, shuffle=True, **kw), DataLoader(val_ds, shuffle=False, **kw)


def make_trainer(max_epochs: int, patience: int, dirpath: str = None,
                 trial: optuna.Trial = None):
    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=patience, mode="min")
    ]
    if dirpath is not None:
        callbacks.append(pl.callbacks.ModelCheckpoint(
            dirpath=dirpath,
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss", save_top_k=1, mode="min",
        ))
    if trial is not None:
        callbacks.append(
            optuna.integration.PyTorchLightningPruningCallback(
                trial, monitor="val_loss"
            )
        )
    cuda = torch.cuda.is_available()
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed" if cuda and torch.cuda.is_bf16_supported() else "16-mixed",
        log_every_n_steps=10,
        enable_progress_bar=True,
        callbacks=callbacks,
    )


def save_results(study: optuna.Study, name: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"best_params_{name}.json")
    with open(path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Paramètres sauvegardés → {path}")

    figs = {
        f"optimization_history_{name}": plot_optimization_history(study),
        f"param_importances_{name}":    plot_param_importances(study),
        f"parallel_coordinate_{name}":  plot_parallel_coordinate(study),
    }
    for fname, fig in figs.items():
        fig_path = os.path.join(OUTPUT_DIR, f"{fname}.html")
        fig.write_html(fig_path)
        print(f"Visualisation → {fig_path}")


# ── Objectif Optuna ───────────────────────────────────────────────────────────

def objective(
    trial:       optuna.Trial,
    obs_dir:     str,
    meta_dir:    str,
    jepa:        HierarchicalJEPAcman,
    jepa_config: Config,
) -> float:

    n_heads  = trial.suggest_categorical("n_heads",  [2, 4])
    n_layers = trial.suggest_int("n_layers", 1, 4)
    lr       = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    # Contrainte têtes / d_model
    if jepa_config.d_model % n_heads != 0:
        raise optuna.TrialPruned()

    dataset = CostDataset(obs_dir, meta_dir, jepa_config)
    if len(dataset) == 0:
        raise optuna.TrialPruned()

    train_loader, val_loader = make_loaders(dataset, batch_size)

    cost_pacman = CostModule(jepa_config.d_model, n_heads, n_layers)
    cost_ghost  = CostModule(jepa_config.d_model, n_heads, n_layers)
    module      = CostModuleModule(jepa, cost_pacman, cost_ghost, lr=lr)
    trainer     = make_trainer(MAX_EPOCHS_TRIALS, EARLY_STOPPING, trial=trial)

    try:
        trainer.fit(module, train_loader, val_loader)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            raise optuna.TrialPruned()
        raise

    val_loss = trainer.callback_metrics.get("val_loss", float("inf")).item()

    del trainer, module, cost_pacman, cost_ghost, train_loader, val_loader, dataset
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return val_loss


# ── Lancement ─────────────────────────────────────────────────────────────────

def run(
    obs_dir:      str,
    meta_dir:     str,
    jepa_weights: str,
    jepa_config:  str,
    n_trials:     int = N_TRIALS,
    storage:      str = None,
):
    jepa, config = load_jepa(jepa_weights, jepa_config)
    config.phase = 2  # CostDataset a besoin des items phase 2

    study = optuna.create_study(
        study_name     = "cost_modules",
        direction      = "minimize",
        sampler        = TPESampler(n_startup_trials=5, multivariate=True, seed=42),
        pruner         = optuna.pruners.NopPruner(),
        storage        = storage,
        load_if_exists = True,
    )

    study.optimize(
        lambda trial: objective(trial, obs_dir, meta_dir, jepa, config),
        n_trials          = n_trials,
        show_progress_bar = True,
    )

    try:
        print(f"\nMeilleur trial : {study.best_trial.number}")
        print(f"Meilleure val_loss : {study.best_value:.4f}")
        print(f"Meilleurs paramètres : {study.best_params}")
    except ValueError:
        raise RuntimeError("Tous les trials ont été pruned.")

    save_results(study, "cost_modules")

    # ── Réentraînement final avec les meilleurs hyperparamètres ──────────────
    print("\nRéentraînement final...")

    best = study.best_params
    n_heads  = best["n_heads"]
    n_layers = best["n_layers"]
    lr       = best["lr"]
    batch_size = best["batch_size"]

    dataset = CostDataset(obs_dir, meta_dir, config)
    train_loader, val_loader = make_loaders(dataset, batch_size)

    cost_pacman = CostModule(config.d_model, n_heads, n_layers)
    cost_ghost  = CostModule(config.d_model, n_heads, n_layers)
    module      = CostModuleModule(jepa, cost_pacman, cost_ghost, lr=lr)

    trainer = make_trainer(
        MAX_EPOCHS, EARLY_STOPPING,
        dirpath=os.path.join(OUTPUT_DIR, "cost"),
    )
    trainer.fit(module, train_loader, val_loader)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(cost_pacman.state_dict(), os.path.join(OUTPUT_DIR, "cost_pacman.pt"))
    torch.save(cost_ghost.state_dict(),  os.path.join(OUTPUT_DIR, "cost_ghost.pt"))
    print(f"Poids sauvegardés → {OUTPUT_DIR}/cost_pacman.pt + cost_ghost.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs-dir",      required=True)
    parser.add_argument("--meta-dir",     required=True)
    parser.add_argument("--jepa-weights", required=True)
    parser.add_argument("--jepa-config",  required=True)
    parser.add_argument("--n-trials",  type=int, default=N_TRIALS)
    parser.add_argument("--storage",   type=str, default=None)
    args = parser.parse_args()

    run(
        obs_dir      = args.obs_dir,
        meta_dir     = args.meta_dir,
        jepa_weights = args.jepa_weights,
        jepa_config  = args.jepa_config,
        n_trials     = args.n_trials,
        storage      = args.storage,
    )