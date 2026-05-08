import os
import json
import torch
import pytorch_lightning as pl
import optuna
import dataclasses
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour,
)
from torch.utils.data import DataLoader, random_split

from config import Config
from data.dataset import PacmanJEPADataset
from lightning.module import HierarchicalJEPAcmanModule


# ── Constantes ────────────────────────────────────────────────────────────────

N_TRIALS       = 100
MAX_EPOCHS     = 100
EARLY_STOPPING = 10
VAL_SPLIT      = 0.1
OUTPUT_DIR     = "optuna_results"


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_loaders(dataset: PacmanJEPADataset, batch_size: int):
    n_val   = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader


def make_trainer(max_epochs: int, patience: int, dirpath: str = None,
                 trial: optuna.Trial = None, enable_progress_bar: bool = False):
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, mode="min"
        )
    ]
    if dirpath is not None:
        callbacks.append(pl.callbacks.ModelCheckpoint(
            dirpath=dirpath,
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        ))
    if trial is not None:
        callbacks.append(
            optuna.integration.PyTorchLightningPruningCallback(
                trial, monitor="val_loss"
            )
        )
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        enable_progress_bar=enable_progress_bar,
        callbacks=callbacks,
    )


def transfer_encoder(src: HierarchicalJEPAcmanModule,
                     dst: HierarchicalJEPAcmanModule):
    """Copie les poids de l'encoder de src vers dst."""
    for attr in ["state_encoder", "tem", "seq_encoder",
                 "target_encoder", "predictor_1"]:
        getattr(dst.model, attr).load_state_dict(
            getattr(src.model, attr).state_dict()
        )


def save_results(study: optuna.Study, best_params: dict, name: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Sauvegarder les paramètres
    path = os.path.join(OUTPUT_DIR, f"best_params_{name}.json")
    with open(path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Paramètres sauvegardés → {path}")

    # Visualisations
    figs = {
        f"optimization_history_{name}" : plot_optimization_history(study),
        f"param_importances_{name}"    : plot_param_importances(study),
        f"parallel_coordinate_{name}"  : plot_parallel_coordinate(study),
    }
    for fname, fig in figs.items():
        fig_path = os.path.join(OUTPUT_DIR, f"{fname}.html")
        fig.write_html(fig_path)
        print(f"Visualisation → {fig_path}")


# ── Étude Phase 1 ─────────────────────────────────────────────────────────────

def objective_phase1(trial: optuna.Trial, data_dir: str) -> float:

    config = Config(
        # Dimensions
        d_model  = trial.suggest_categorical("d_model", [64, 128, 256]),
        seq_len  = trial.suggest_categorical("seq_len", [64, 128, 256]),

        # Horizons — fixés ici car ils affectent aussi phase 2
        k = trial.suggest_categorical("k", [4, 8, 16]),
        j = trial.suggest_categorical("j", [16, 32, 64]),

        # State encoder
        n_heads_state  = trial.suggest_categorical("n_heads_state", [2, 4]),
        n_layers_state = trial.suggest_int("n_layers_state", 1, 4),

        # Sequence encoder
        n_heads_seq    = trial.suggest_categorical("n_heads_seq", [2, 4, 8]),
        n_layers_seq   = trial.suggest_int("n_layers_seq", 2, 6),

        # Predictor niveau 1
        n_heads_pred_1  = trial.suggest_categorical("n_heads_pred_1", [2, 4]),
        n_layers_pred_1 = trial.suggest_int("n_layers_pred_1", 1, 4),

        # Entraînement
        ema_momentum = trial.suggest_float("ema_momentum", 0.99, 0.999),
        lr           = trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        batch_size   = trial.suggest_categorical("batch_size", [32, 64, 128]),

        phase = 1,
    )

    # Contraintes
    if config.j % config.k != 0:
        raise optuna.TrialPruned()
    for n_heads in [config.n_heads_state, config.n_heads_seq, config.n_heads_pred_1]:
        if config.d_model % n_heads != 0:
            raise optuna.TrialPruned()

    dataset = PacmanJEPADataset(data_dir, config)
    train_loader, val_loader = make_loaders(dataset, config.batch_size)

    module  = HierarchicalJEPAcmanModule(config)
    trainer = make_trainer(
        max_epochs=MAX_EPOCHS, patience=EARLY_STOPPING, trial=trial
    )
    trainer.fit(module, train_loader, val_loader)

    return trainer.callback_metrics.get("val_loss", float("inf")).item()


def run_phase1(data_dir: str, storage: str = None) -> tuple:
    """
    Lance l'optimisation phase 1.
    Retourne (best_config, best_module).
    """
    study = optuna.create_study(
        study_name     = "jepa_phase1",
        direction      = "minimize",
        sampler        = TPESampler(
            n_startup_trials = 10,
            multivariate     = True,
            seed             = 42,
        ),
        pruner         = HyperbandPruner(
            min_resource     = 5,
            max_resource     = MAX_EPOCHS,
            reduction_factor = 3,
        ),
        storage        = storage,
        load_if_exists = True,
    )

    study.optimize(
        lambda trial: objective_phase1(trial, data_dir),
        n_trials          = N_TRIALS,
        show_progress_bar = True,
    )

    print(f"\n[Phase 1] Meilleur trial : {study.best_trial.number}")
    print(f"[Phase 1] Meilleure val_loss : {study.best_value:.4f}")

    save_results(study, study.best_params, "phase1")

    # Réentraînement final phase 1 avec early stopping sans borne artificielle
    print("\n[Phase 1] Réentraînement final...")

    best_config = Config(**{
        k: v for k, v in study.best_params.items()
        if k in Config.__dataclass_fields__
    }, phase=1)

    dataset = PacmanJEPADataset(data_dir, best_config)
    train_loader, val_loader = make_loaders(dataset, best_config.batch_size)

    best_module = HierarchicalJEPAcmanModule(best_config)
    trainer     = make_trainer(
        max_epochs           = MAX_EPOCHS,
        patience             = EARLY_STOPPING,
        dirpath              = os.path.join(OUTPUT_DIR, "phase1"),
        enable_progress_bar  = True,
    )
    trainer.fit(best_module, train_loader, val_loader)

    # Sauvegarder les poids phase 1
    weights_path = os.path.join(OUTPUT_DIR, "phase1", "encoder.pt")
    torch.save(best_module.model.state_dict(), weights_path)
    print(f"[Phase 1] Poids sauvegardés → {weights_path}")

    return best_config, best_module


# ── Étude Phase 2 ─────────────────────────────────────────────────────────────

def objective_phase2(trial: optuna.Trial, data_dir: str,
                     phase1_config: Config,
                     phase1_module: HierarchicalJEPAcmanModule) -> float:

    config = Config(
        # Reprendre tous les paramètres de phase 1 — encoder gelé
        d_model        = phase1_config.d_model,
        seq_len        = phase1_config.seq_len,
        k              = phase1_config.k,
        j              = phase1_config.j,
        n_heads_state  = phase1_config.n_heads_state,
        n_layers_state = phase1_config.n_layers_state,
        n_heads_seq    = phase1_config.n_heads_seq,
        n_layers_seq   = phase1_config.n_layers_seq,
        n_heads_pred_1 = phase1_config.n_heads_pred_1,
        n_layers_pred_1= phase1_config.n_layers_pred_1,
        ema_momentum   = phase1_config.ema_momentum,

        # Optimiser uniquement les predicteurs 2 et 3
        n_heads_pred_2  = trial.suggest_categorical("n_heads_pred_2", [2, 4]),
        n_layers_pred_2 = trial.suggest_int("n_layers_pred_2", 1, 4),
        n_heads_pred_3  = trial.suggest_categorical("n_heads_pred_3", [2, 4]),
        n_layers_pred_3 = trial.suggest_int("n_layers_pred_3", 1, 4),

        # Lr et batch_size peuvent être différents en phase 2
        lr         = trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128]),

        phase = 2,
    )

    # Contraintes têtes
    for n_heads in [config.n_heads_pred_2, config.n_heads_pred_3]:
        if config.d_model % n_heads != 0:
            raise optuna.TrialPruned()

    dataset = PacmanJEPADataset(data_dir, config)
    train_loader, val_loader = make_loaders(dataset, config.batch_size)

    module = HierarchicalJEPAcmanModule(config)
    transfer_encoder(phase1_module, module)   # encoder gelé dans on_fit_start

    trainer = make_trainer(
        max_epochs=MAX_EPOCHS, patience=EARLY_STOPPING, trial=trial
    )
    trainer.fit(module, train_loader, val_loader)

    return trainer.callback_metrics.get("val_loss", float("inf")).item()


def run_phase2(data_dir: str, phase1_config: Config,
               phase1_module: HierarchicalJEPAcmanModule,
               storage: str = None) -> HierarchicalJEPAcmanModule:
    """
    Lance l'optimisation phase 2.
    Retourne le meilleur module complet.
    """
    study = optuna.create_study(
        study_name     = "jepa_phase2",
        direction      = "minimize",
        sampler        = TPESampler(
            n_startup_trials = 10,
            multivariate     = True,
            seed             = 42,
        ),
        pruner         = HyperbandPruner(
            min_resource     = 5,
            max_resource     = MAX_EPOCHS,
            reduction_factor = 3,
        ),
        storage        = storage,
        load_if_exists = True,
    )

    study.optimize(
        lambda trial: objective_phase2(
            trial, data_dir, phase1_config, phase1_module
        ),
        n_trials          = N_TRIALS,
        show_progress_bar = True,
    )

    print(f"\n[Phase 2] Meilleur trial : {study.best_trial.number}")
    print(f"[Phase 2] Meilleure val_loss : {study.best_value:.4f}")

    save_results(study, study.best_params, "phase2")

    # Réentraînement final phase 2
    print("\n[Phase 2] Réentraînement final...")

    params = dataclasses.asdict(phase1_config)
    params.update({
        "n_heads_pred_2"  : study.best_params["n_heads_pred_2"],
        "n_layers_pred_2" : study.best_params["n_layers_pred_2"],
        "n_heads_pred_3"  : study.best_params["n_heads_pred_3"],
        "n_layers_pred_3" : study.best_params["n_layers_pred_3"],
        "lr"              : study.best_params["lr"],
        "batch_size"      : study.best_params["batch_size"],
        "phase"           : 2,
    })
    best_config = Config(**params)

    dataset = PacmanJEPADataset(data_dir, best_config)
    train_loader, val_loader = make_loaders(dataset, best_config.batch_size)

    best_module = HierarchicalJEPAcmanModule(best_config)
    transfer_encoder(phase1_module, best_module)

    trainer = make_trainer(
        max_epochs          = MAX_EPOCHS,
        patience            = EARLY_STOPPING,
        dirpath             = os.path.join(OUTPUT_DIR, "phase2"),
        enable_progress_bar = True,
    )
    trainer.fit(best_module, train_loader, val_loader)

    # Sauvegarder poids + architecture complète
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    weights_path = os.path.join(OUTPUT_DIR, "best_model.pt")
    torch.save(best_module.model.state_dict(), weights_path)
    print(f"[Phase 2] Poids sauvegardés → {weights_path}")

    arch_path = os.path.join(OUTPUT_DIR, "best_config_full.json")
    with open(arch_path, "w") as f:
        json.dump(best_config.__dict__, f, indent=2)
    print(f"[Phase 2] Architecture sauvegardée → {arch_path}")

    return best_module


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--n_trials", type=int, default=N_TRIALS)
    parser.add_argument("--storage",  type=str, default=None,
                        help="ex: sqlite:///optuna.db")
    args = parser.parse_args()

    N_TRIALS = args.n_trials

    # ── Étape 1 — optimiser et entraîner phase 1 ─────────────────────────────
    phase1_config, phase1_module = run_phase1(args.data_dir, args.storage)

    # ── Étape 2 — optimiser et entraîner phase 2 ─────────────────────────────
    run_phase2(args.data_dir, phase1_config, phase1_module, args.storage)