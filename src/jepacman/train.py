import os
import json
import pickle

import torch
import pytorch_lightning as pl
import optuna
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

N_TRIALS       = 100     # nombre d'essais optuna
MAX_EPOCHS     = 100     # borne max même sans early stopping
EARLY_STOPPING = 10      # patience early stopping
VAL_SPLIT      = 0.1
OUTPUT_DIR     = "optuna_results"


# ── Objective ─────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, data_dir: str) -> float:
    """
    Retourne la meilleure val_loss phase 2 pour un jeu de paramètres.
    Optuna minimise cette valeur.
    """

    # ── Suggérer les hyperparamètres ─────────────────────────────────────────
    config = Config(
        # Dimensions
        d_model  = trial.suggest_categorical("d_model", [64, 128, 256]),
        seq_len  = trial.suggest_categorical("seq_len", [64, 128, 256]),

        # Horizons hiérarchiques
        k = trial.suggest_categorical("k", [4, 8, 16]),
        j = trial.suggest_categorical("j", [16, 32, 64]),

        # State encoder (ViT)
        n_heads_state  = trial.suggest_categorical("n_heads_state", [2, 4]),
        n_layers_state = trial.suggest_int("n_layers_state", 1, 4),

        # Sequence encoder
        n_heads_seq    = trial.suggest_categorical("n_heads_seq", [2, 4, 8]),
        n_layers_seq   = trial.suggest_int("n_layers_seq", 2, 6),

        # Predictor niveau 1
        n_heads_pred_1  = trial.suggest_categorical("n_heads_pred_1", [2, 4]),
        n_layers_pred_1 = trial.suggest_int("n_layers_pred_1", 1, 4),

        # Predictor niveau 2
        n_heads_pred_2  = trial.suggest_categorical("n_heads_pred_2", [2, 4]),
        n_layers_pred_2 = trial.suggest_int("n_layers_pred_2", 1, 4),

        # Predictor niveau 3
        n_heads_pred_3  = trial.suggest_categorical("n_heads_pred_3", [2, 4]),
        n_layers_pred_3 = trial.suggest_int("n_layers_pred_3", 1, 4),

        # JEPA
        ema_momentum = trial.suggest_float("ema_momentum", 0.99, 0.999),

        # Entraînement
        lr         = trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128]),
    )

    # Contrainte : j divisible par k
    if config.j % config.k != 0:
        raise optuna.TrialPruned()

    # Contrainte : d_model divisible par toutes les têtes
    for n_heads in [
        config.n_heads_state, config.n_heads_seq,
        config.n_heads_pred_1, config.n_heads_pred_2, config.n_heads_pred_3
    ]:
        if config.d_model % n_heads != 0:
            raise optuna.TrialPruned()

    # ── Dataset ───────────────────────────────────────────────────────────────
    config.phase = 1
    dataset      = PacmanJEPADataset(data_dir, config)
    n_val        = int(len(dataset) * VAL_SPLIT)
    n_train      = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # ── Callback Optuna pruning ───────────────────────────────────────────────
    pruning_callback = optuna.integration.PyTorchLightningPruningCallback(
        trial, monitor="val_loss"
    )

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    config.phase  = 1
    module_phase1 = HierarchicalJEPAcmanModule(config)

    trainer_phase1 = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        enable_progress_bar=False,   # silencieux pendant l'optim
        callbacks=[
            pruning_callback,
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING,
                mode="min",
            ),
        ],
    )

    trainer_phase1.fit(module_phase1, train_loader, val_loader)

    # Pruner — abandonner si phase 1 n'est pas prometteuse
    if trainer_phase1.callback_metrics.get("val_loss", float("inf")) > 1.0:
        raise optuna.TrialPruned()

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    config.phase  = 2
    dataset.config.phase = 2
    module_phase2 = HierarchicalJEPAcmanModule(config)

    # Copier les poids de phase 1
    for attr in ["state_encoder", "tem", "seq_encoder", "target_encoder", "predictor_1"]:
        getattr(module_phase2.model, attr).load_state_dict(
            getattr(module_phase1.model, attr).state_dict()
        )

    trainer_phase2 = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        enable_progress_bar=False,
        callbacks=[
            pruning_callback,
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING,
                mode="min",
            ),
        ],
    )

    trainer_phase2.fit(module_phase2, train_loader, val_loader)

    val_loss = trainer_phase2.callback_metrics.get("val_loss", float("inf"))

    # ── Sauvegarder si meilleur trial ────────────────────────────────────────
    trial.set_user_attr("val_loss_phase1",
        trainer_phase1.callback_metrics.get("val_loss", float("inf")).item()
    )

    return val_loss.item()


# ── Sauvegarder le meilleur modèle ───────────────────────────────────────────

def save_best(study: optuna.Study, data_dir: str):
    """
    Réentraîne avec les meilleurs paramètres et sauvegarde
    les poids + l'architecture.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    best_params = study.best_params
    print(f"\nMeilleurs paramètres : {best_params}")

    # Reconstruire le config
    config = Config(**{
        k: v for k, v in best_params.items()
        if k in Config.__dataclass_fields__
    })

    # Sauvegarder l'architecture
    arch_path = os.path.join(OUTPUT_DIR, "best_config.json")
    with open(arch_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Architecture sauvegardée → {arch_path}")

    # Réentraîner avec early stopping sans borne artificielle
    dataset = PacmanJEPADataset(data_dir, config)
    n_val   = int(len(dataset) * VAL_SPLIT)
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=config.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING, mode="min"
    )

    # Phase 1
    config.phase  = 1
    module_phase1 = HierarchicalJEPAcmanModule(config)
    trainer1      = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        callbacks=[
            early_stop,
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(OUTPUT_DIR, "phase1"),
                filename="best-phase1-{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                save_top_k=1,
                mode="min",
            ),
        ],
    )
    trainer1.fit(module_phase1, train_loader, val_loader)

    # Phase 2
    config.phase  = 2
    module_phase2 = HierarchicalJEPAcmanModule(config)
    for attr in ["state_encoder", "tem", "seq_encoder", "target_encoder", "predictor_1"]:
        getattr(module_phase2.model, attr).load_state_dict(
            getattr(module_phase1.model, attr).state_dict()
        )

    trainer2 = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss", patience=EARLY_STOPPING, mode="min"
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(OUTPUT_DIR, "phase2"),
                filename="best-phase2-{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                save_top_k=1,
                mode="min",
            ),
        ],
    )
    trainer2.fit(module_phase2, train_loader, val_loader)

    # Sauvegarder les poids complets
    weights_path = os.path.join(OUTPUT_DIR, "best_model.pt")
    torch.save(module_phase2.model.state_dict(), weights_path)
    print(f"Poids sauvegardés → {weights_path}")


# ── Visualisations ────────────────────────────────────────────────────────────

def visualize(study: optuna.Study):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    figs = {
        "optimization_history"  : plot_optimization_history(study),
        "param_importances"     : plot_param_importances(study),
        "parallel_coordinate"   : plot_parallel_coordinate(study),
        "contour_lr_d_model"    : plot_contour(study, params=["lr", "d_model"]),
    }

    for name, fig in figs.items():
        path = os.path.join(OUTPUT_DIR, f"{name}.html")
        fig.write_html(path)
        print(f"Visualisation → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Dossier contenant les sessions converties")
    parser.add_argument("--n_trials", type=int, default=N_TRIALS)
    parser.add_argument("--study_name", type=str, default="hierarchical_jepa")
    parser.add_argument("--storage", type=str, default=None,
                        help="URL SQLite ou PostgreSQL pour persister l'étude "
                             "ex: sqlite:///optuna.db")
    args = parser.parse_args()

    # TPE — meilleur algorithme de recherche bayésien d'Optuna
    # HyperbandPruner — arrête tôt les trials non prometteurs
    study = optuna.create_study(
        study_name  = args.study_name,
        direction   = "minimize",
        sampler     = TPESampler(
            n_startup_trials = 10,   # exploration aléatoire avant TPE
            multivariate     = True, # modélise les corrélations entre paramètres
            seed             = 42,
        ),
        pruner  = HyperbandPruner(
            min_resource  = 5,    # évaluer au moins 5 epochs avant pruning
            max_resource  = MAX_EPOCHS,
            reduction_factor = 3,
        ),
        storage = args.storage,
        load_if_exists = True,   # reprendre une étude existante
    )

    study.optimize(
        lambda trial: objective(trial, args.data_dir),
        n_trials   = args.n_trials,
        timeout    = None,
        show_progress_bar = True,
    )

    print(f"\nMeilleur trial : {study.best_trial.number}")
    print(f"Meilleure val_loss : {study.best_value:.4f}")

    visualize(study)
    save_best(study, args.data_dir)