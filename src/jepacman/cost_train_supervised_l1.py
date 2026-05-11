"""
Réentraînement supervisé des modules de coût pour le JEPA niveau-1 (chaîne de 5 pas).

Réutilise l'architecture choisie par Optuna (best_params_cost_modules.json)
et réentraîne sur le dataset L1.

Usage :
    python -m src.jepacman.cost_train_supervised_l1 \\
        --obs-dir   jepa_numpy \\
        --meta-dir  jepa_meta \\
        --jepa-weights optuna_results/best_model.pt \\
        --jepa-config  optuna_results/best_config_full.json
"""

import os
import json
import argparse
import dataclasses

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from .config import Config
from .models.jepa import HierarchicalJEPAcman
from .models.cost_module import CostModule
from .data.cost_dataset import CostDatasetL1
from .lightning.cost_module_l1_pl import CostModuleL1Module

MAX_EPOCHS     = 50
EARLY_STOPPING = 7
VAL_SPLIT      = 0.1
OUTPUT_DIR     = "optuna_results"


def load_jepa(jepa_weights: str, jepa_config: str):
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


def run(
    obs_dir:      str,
    meta_dir:     str,
    jepa_weights: str,
    jepa_config:  str,
    lr:           float = 3e-4,
    batch_size:   int   = 64,
):
    jepa, config = load_jepa(jepa_weights, jepa_config)

    # Charger l'architecture choisie par Optuna
    params_path = os.path.join(OUTPUT_DIR, "best_params_cost_modules.json")
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = json.load(f)
        n_heads  = params.get("n_heads",  2)
        n_layers = params.get("n_layers", 2)
        lr       = params.get("lr", lr)
        batch_size = params.get("batch_size", batch_size)
        print(f"Architecture chargée : n_heads={n_heads}, n_layers={n_layers}, lr={lr:.2e}")
    else:
        n_heads, n_layers = 2, 2
        print("⚠ best_params_cost_modules.json introuvable — architecture par défaut")

    dataset = CostDatasetL1(obs_dir, meta_dir, config)
    print(f"{len(dataset)} items dans le dataset L1")
    if len(dataset) == 0:
        raise RuntimeError("Dataset vide — vérifier obs-dir et meta-dir.")

    n_val   = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    cuda = torch.cuda.is_available()
    kw = dict(batch_size=batch_size, num_workers=4,
               pin_memory=cuda, persistent_workers=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)

    cost_pacman = CostModule(config.d_model, n_heads, n_layers)
    cost_ghost  = CostModule(config.d_model, n_heads, n_layers)
    module      = CostModuleL1Module(jepa, cost_pacman, cost_ghost, lr=lr)

    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING, mode="min"),
        pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(OUTPUT_DIR, "cost_l1"),
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss", save_top_k=1, mode="min",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed" if cuda and torch.cuda.is_bf16_supported() else "16-mixed",
        log_every_n_steps=10,
        enable_progress_bar=True,
        callbacks=callbacks,
    )

    trainer.fit(module, train_loader, val_loader)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(cost_pacman.state_dict(), os.path.join(OUTPUT_DIR, "cost_pacman_l1.pt"))
    torch.save(cost_ghost.state_dict(),  os.path.join(OUTPUT_DIR, "cost_ghost_l1.pt"))
    print(f"Poids sauvegardés → {OUTPUT_DIR}/cost_pacman_l1.pt + cost_ghost_l1.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs-dir",      required=True)
    parser.add_argument("--meta-dir",     required=True)
    parser.add_argument("--jepa-weights", required=True)
    parser.add_argument("--jepa-config",  required=True)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--batch-size",   type=int,   default=64)
    args = parser.parse_args()

    run(
        obs_dir      = args.obs_dir,
        meta_dir     = args.meta_dir,
        jepa_weights = args.jepa_weights,
        jepa_config  = args.jepa_config,
        lr           = args.lr,
        batch_size   = args.batch_size,
    )
