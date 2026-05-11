"""
Tests des modules de coût JEPA-Pacman.
=======================================
Couvre :
  1. CostModule        — forward, shapes
  2. CostDataset       — chargement, shapes, calcul du coût
  3. CostModuleModule  — _step avec batch synthétique
  4. Mini-entraînement — 2 epochs sur données synthétiques

Pour lancer :
    pytest tests/test_cost_modules.py -v
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.jepacman.config import Config
from src.jepacman.models.cost_module import CostModule
from src.jepacman.models.jepa import HierarchicalJEPAcman
from src.jepacman.data.cost_dataset import CostDataset
from src.jepacman.lightning.cost_module_pl import CostModuleModule


# ── Config légère pour les tests ─────────────────────────────────────────────

MINI_CFG = Config(
    case_dim=8,
    n_actions=5,
    d_model=32,
    seq_len=8,
    max_visible_cells=81,
    k=2,
    j=4,
    n_heads_state=2,  n_layers_state=1,
    n_heads_seq=2,    n_layers_seq=1,
    n_heads_pred_1=2, n_layers_pred_1=1,
    n_heads_pred_2=2, n_layers_pred_2=1,
    n_heads_pred_3=2, n_layers_pred_3=1,
    ema_momentum=0.99,
    lr=1e-3,
    batch_size=4,
    phase=2,
    stride=1,
)

B = 4
T = MINI_CFG.seq_len
D = MINI_CFG.d_model
N = MINI_CFG.max_visible_cells   # 81 cellules
J = MINI_CFG.j


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_jepa(cfg: Config = MINI_CFG) -> HierarchicalJEPAcman:
    return HierarchicalJEPAcman(
        case_dim=cfg.case_dim,
        n_actions=cfg.n_actions,
        d_model=cfg.d_model,
        seq_len=cfg.seq_len,
        n_heads_state=cfg.n_heads_state,   n_layers_state=cfg.n_layers_state,
        n_heads_seq=cfg.n_heads_seq,       n_layers_seq=cfg.n_layers_seq,
        n_heads_pred_1=cfg.n_heads_pred_1, n_layers_pred_1=cfg.n_layers_pred_1,
        n_heads_pred_2=cfg.n_heads_pred_2, n_layers_pred_2=cfg.n_layers_pred_2,
        n_heads_pred_3=cfg.n_heads_pred_3, n_layers_pred_3=cfg.n_layers_pred_3,
        ema_momentum=cfg.ema_momentum,
        k=cfg.k,
        j=cfg.j,
    ).eval()


def make_synthetic_batch(cfg: Config = MINI_CFG, batch_size: int = B):
    """Construit un batch synthétique identique à ce que retourne CostDataset."""
    cases_seq      = torch.zeros(batch_size, T, N, cfg.case_dim)
    pos_seq        = torch.zeros(batch_size, T, N, 2)
    masks_seq      = torch.zeros(batch_size, T, N, dtype=torch.bool)
    actions_seq    = torch.zeros(batch_size, T, dtype=torch.long)
    actions_future = torch.zeros(batch_size, cfg.j, dtype=torch.long)
    cost_real      = torch.rand(batch_size)
    agent_type     = torch.randint(0, 2, (batch_size,))
    return (cases_seq, pos_seq, masks_seq, actions_seq,
            actions_future, cost_real, agent_type)


def write_synthetic_npz(tmp_obs: str, tmp_meta: str, cfg: Config,
                        n_frames: int = 30, n_agents: int = 5):
    """Crée un fichier .npz d'observations et un _meta.npz synthétiques."""
    os.makedirs(tmp_obs,  exist_ok=True)
    os.makedirs(tmp_meta, exist_ok=True)

    fname = "session_test.npz"

    # Observations
    cases     = np.zeros((n_frames, n_agents, N, cfg.case_dim), dtype=np.float32)
    positions = np.zeros((n_frames, n_agents, N, 2),            dtype=np.float32)
    pad_masks = np.zeros((n_frames, n_agents, N),               dtype=bool)
    actions   = np.zeros((n_frames, n_agents),                  dtype=np.int8)
    np.savez_compressed(
        os.path.join(tmp_obs, fname),
        cases=cases, positions=positions,
        pad_masks=pad_masks, actions=actions,
    )

    # Métadonnées
    n_pellets            = np.arange(n_frames, 0, -1, dtype=np.int32)
    outcome_eaten        = np.bool_(False)
    ghost_dist_to_pacman = np.ones((n_frames, 4), dtype=np.int16) * 3
    np.savez_compressed(
        os.path.join(tmp_meta, fname),
        n_pellets=n_pellets,
        outcome_eaten=outcome_eaten,
        ghost_dist_to_pacman=ghost_dist_to_pacman,
    )

    return os.path.join(tmp_obs, fname), os.path.join(tmp_meta, fname)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CostModule — forward et shapes
# ══════════════════════════════════════════════════════════════════════════════

class TestCostModule:

    def test_output_shape(self):
        model  = CostModule(d_model=D, n_heads=2, n_layers=1)
        out    = model(torch.randn(B, T, D), torch.randn(B, D), torch.randn(B, D))
        assert out.shape == (B,), f"Attendu ({B},), obtenu {out.shape}"

    def test_output_scalar_batch1(self):
        model = CostModule(d_model=D, n_heads=2, n_layers=1)
        out   = model(torch.randn(1, T, D), torch.randn(1, D), torch.randn(1, D))
        assert out.shape == (1,)

    def test_gradient_flows(self):
        model  = CostModule(d_model=D, n_heads=2, n_layers=1)
        loss   = model(torch.randn(B, T, D), torch.randn(B, D), torch.randn(B, D)).mean()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_gradient_on_gj(self):
        """Le gradient remonte jusqu'à g_j (nécessaire pour plan_with_cost)."""
        model = CostModule(d_model=D, n_heads=2, n_layers=1)
        g_j   = torch.randn(B, D, requires_grad=True)
        loss  = model(torch.randn(B, T, D), torch.randn(B, D), g_j).mean()
        loss.backward()
        assert g_j.grad is not None and g_j.grad.abs().sum() > 0

    def test_different_seq_lengths(self):
        model = CostModule(d_model=D, n_heads=2, n_layers=1)
        for t in [1, 4, 16]:
            out = model(torch.randn(B, t, D), torch.randn(B, D), torch.randn(B, D))
            assert out.shape == (B,)

    def test_no_nan(self):
        model = CostModule(d_model=D, n_heads=2, n_layers=1)
        out   = model(torch.randn(B, T, D), torch.randn(B, D), torch.randn(B, D))
        assert not torch.isnan(out).any()


# ══════════════════════════════════════════════════════════════════════════════
# 2. CostDataset — chargement, shapes, calcul du coût
# ══════════════════════════════════════════════════════════════════════════════

class TestCostDataset:

    @pytest.fixture
    def dataset(self, tmp_path):
        obs_dir  = str(tmp_path / "obs")
        meta_dir = str(tmp_path / "meta")
        write_synthetic_npz(obs_dir, meta_dir, MINI_CFG, n_frames=30, n_agents=5)
        return CostDataset(obs_dir, meta_dir, MINI_CFG)

    def test_not_empty(self, dataset):
        assert len(dataset) > 0

    def test_item_count(self, tmp_path):
        obs_dir  = str(tmp_path / "obs")
        meta_dir = str(tmp_path / "meta")
        write_synthetic_npz(obs_dir, meta_dir, MINI_CFG, n_frames=30, n_agents=5)
        ds = CostDataset(obs_dir, meta_dir, MINI_CFG)
        # max_t = 30 - 1 - j = 25 ; t in range(0, 26, stride=1) → 26 items × 5 agents
        assert len(ds) == 26 * 5

    def test_item_shapes(self, dataset):
        cfg = MINI_CFG
        (cases_seq, pos_seq, masks_seq, actions_seq,
         actions_future_j, cost_real, agent_type) = dataset[0]

        assert cases_seq.shape        == (cfg.seq_len, N, cfg.case_dim)
        assert pos_seq.shape          == (cfg.seq_len, N, 2)
        assert masks_seq.shape        == (cfg.seq_len, N)
        assert actions_seq.shape      == (cfg.seq_len,)
        assert actions_future_j.shape == (cfg.j,)
        assert cost_real.shape        == ()
        assert agent_type.shape       == ()

    def test_cost_pacman_death(self, tmp_path):
        """Pac-Man mort → coût = j * 10."""
        obs_dir  = str(tmp_path / "obs")
        meta_dir = str(tmp_path / "meta")
        n_frames = 30
        os.makedirs(obs_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
        fname = "session_test.npz"

        cases     = np.zeros((n_frames, 5, N, MINI_CFG.case_dim), dtype=np.float32)
        positions = np.zeros((n_frames, 5, N, 2), dtype=np.float32)
        pad_masks = np.zeros((n_frames, 5, N), dtype=bool)
        actions   = np.zeros((n_frames, 5), dtype=np.int8)
        np.savez_compressed(os.path.join(obs_dir, fname),
                            cases=cases, positions=positions,
                            pad_masks=pad_masks, actions=actions)

        np.savez_compressed(
            os.path.join(meta_dir, fname),
            n_pellets=np.arange(n_frames, 0, -1, dtype=np.int32),
            outcome_eaten=np.bool_(True),
            ghost_dist_to_pacman=np.ones((n_frames, 4), dtype=np.int16),
        )
        ds = CostDataset(obs_dir, meta_dir, MINI_CFG)
        last_t = n_frames - 1 - MINI_CFG.j
        item_idx = next(
            i for i, (_, _, aid, t) in enumerate(ds.items)
            if aid == 0 and t == last_t
        )
        _, _, _, _, _, cost_real, agent_type = ds[item_idx]
        assert agent_type.item() == 0
        assert cost_real.item() == pytest.approx(MINI_CFG.j * 10)

    def test_cost_ghost(self, dataset):
        """dist_t=3, dist_tj=3 → coût fantôme = 2*3 - 3 = 3."""
        ghost_idx = next(i for i, (_, _, aid, _) in enumerate(dataset.items) if aid > 0)
        *_, cost_real, agent_type = dataset[ghost_idx]
        assert agent_type.item() == 1
        assert cost_real.item() == pytest.approx(3.0)

    def test_dataloader_collation(self, dataset):
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        batch  = next(iter(loader))
        assert batch[0].shape == (4, MINI_CFG.seq_len, N, MINI_CFG.case_dim)


# ══════════════════════════════════════════════════════════════════════════════
# 3. CostModuleModule — step avec batch synthétique
# ══════════════════════════════════════════════════════════════════════════════

class TestCostModuleModule:

    @pytest.fixture
    def module(self):
        jepa        = make_jepa()
        cost_pacman = CostModule(D, n_heads=2, n_layers=1)
        cost_ghost  = CostModule(D, n_heads=2, n_layers=1)
        m = CostModuleModule(jepa, cost_pacman, cost_ghost, lr=1e-3)
        for p in m.jepa.parameters():
            p.requires_grad = False
        m.jepa.eval()
        return m

    def test_step_scalar(self, module):
        loss = module._step(make_synthetic_batch())
        assert loss.shape == ()

    def test_step_positive(self, module):
        loss = module._step(make_synthetic_batch())
        assert loss.item() >= 0

    def test_step_no_nan(self, module):
        loss = module._step(make_synthetic_batch())
        assert not torch.isnan(loss)

    def test_pacman_only_batch(self, module):
        batch     = list(make_synthetic_batch())
        batch[-1] = torch.zeros(B, dtype=torch.long)
        assert not torch.isnan(module._step(tuple(batch)))

    def test_ghost_only_batch(self, module):
        batch     = list(make_synthetic_batch())
        batch[-1] = torch.ones(B, dtype=torch.long)
        assert not torch.isnan(module._step(tuple(batch)))

    def test_jepa_frozen(self, module):
        loss = module._step(make_synthetic_batch())
        loss.backward()
        for name, p in module.jepa.named_parameters():
            assert p.grad is None or p.grad.abs().sum() == 0, \
                f"Poids JEPA '{name}' a reçu un gradient"

    def test_cost_modules_get_gradient(self, module):
        loss = module._step(make_synthetic_batch())
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in list(module.cost_pacman.parameters())
                     + list(module.cost_ghost.parameters())
        )
        assert has_grad

    def test_configure_optimizers(self, module):
        opt = module.configure_optimizers()
        opt_params  = set(id(p) for g in opt.param_groups for p in g["params"])
        cost_params = set(id(p) for p in list(module.cost_pacman.parameters())
                                          + list(module.cost_ghost.parameters()))
        jepa_params = set(id(p) for p in module.jepa.parameters())

        assert cost_params.issubset(opt_params), \
            "Des paramètres cost_module manquent dans l'optimiseur"
        assert jepa_params.isdisjoint(opt_params), \
            "Des paramètres JEPA sont dans l'optimiseur"


# ══════════════════════════════════════════════════════════════════════════════
# 4. Mini-entraînement — 2 epochs sur données synthétiques
# ══════════════════════════════════════════════════════════════════════════════

class TestMiniTraining:

    def test_2_epoch_training(self, tmp_path):
        """val_loss enregistrée et finie après 2 epochs."""
        obs_dir  = str(tmp_path / "obs")
        meta_dir = str(tmp_path / "meta")
        write_synthetic_npz(obs_dir, meta_dir, MINI_CFG, n_frames=30, n_agents=5)

        dataset = CostDataset(obs_dir, meta_dir, MINI_CFG)
        n_val   = max(1, int(len(dataset) * 0.2))
        n_train = len(dataset) - n_val
        from torch.utils.data import random_split
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=4, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=4, num_workers=0)

        module = CostModuleModule(
            make_jepa(),
            CostModule(D, n_heads=2, n_layers=1),
            CostModule(D, n_heads=2, n_layers=1),
            lr=1e-3,
        )
        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            devices=1,
            precision="32-true",
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(module, train_loader, val_loader)

        val_loss = trainer.callback_metrics.get("val_loss")
        assert val_loss is not None
        assert torch.isfinite(val_loss)

    def test_weights_change_after_training(self, tmp_path):
        """cost_pacman change, JEPA reste identique."""
        obs_dir  = str(tmp_path / "obs")
        meta_dir = str(tmp_path / "meta")
        write_synthetic_npz(obs_dir, meta_dir, MINI_CFG, n_frames=30, n_agents=5)

        dataset = CostDataset(obs_dir, meta_dir, MINI_CFG)
        loader  = DataLoader(dataset, batch_size=4, num_workers=0)

        jepa        = make_jepa()
        cost_pacman = CostModule(D, n_heads=2, n_layers=1)
        cost_ghost  = CostModule(D, n_heads=2, n_layers=1)

        jepa_before = {k: v.clone() for k, v in jepa.state_dict().items()}
        pac_before  = {k: v.clone() for k, v in cost_pacman.state_dict().items()}

        module  = CostModuleModule(jepa, cost_pacman, cost_ghost, lr=1e-3)
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            precision="32-true",
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(module, loader, loader)

        for k, v in jepa.state_dict().items():
            assert torch.allclose(v, jepa_before[k]), \
                f"Poids JEPA '{k}' ont changé"

        changed = any(
            not torch.allclose(v, pac_before[k])
            for k, v in cost_pacman.state_dict().items()
        )
        assert changed, "Aucun poids de cost_pacman n'a changé"