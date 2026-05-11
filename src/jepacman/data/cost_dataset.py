"""
Dataset pour l'entraînement supervisé des modules de coût.

Charge les observations JEPA (.npz) + les métadonnées de jeu (_meta.npz)
et calcule les coûts réels pour chaque item.

Fonctions de coût
-----------------
Pac-Man (agent_id = 0) :
    cost = j * 10                       si Pac-Man est mangé entre t et t+j
    cost = j - pellets_mangés(t, t+j)   sinon

Fantôme (agent_id = 1..4) :
    cost = 2 * dist_{t+j}(pacman) - dist_t(pacman)
    (distance Manhattan entre le fantôme et Pac-Man)

Usage :
    dataset = CostDataset("jepa_numpy", "jepa_meta", config)
"""

import os
import glob
import numpy as np
import torch
from collections import OrderedDict
from typing import List, Tuple
from torch.utils.data import Dataset

from ..config import Config

CHAIN_LEN = 5   # horizon fixe pour CostDatasetL1


class CostDataset(Dataset):
    """
    Items retournés (tuples) :
        cases_seq, pos_seq, masks_seq, actions_seq,    # contexte
        n_real,
        cases_next, pos_next, pad_next, action_next,   # t+1
        cases_tk, pos_tk, pad_tk, actions_future_k,    # t+k
        cases_tj, pos_tj, pad_tj, actions_future_j,    # t+j
        cost_real,   # float32 scalaire — coût réel calculé
        agent_type,  # long 0=pacman 1=fantôme
    """

    def __init__(self, obs_dir: str, meta_dir: str, config: Config):
        self.seq_len  = config.seq_len
        self.k        = config.k
        self.j        = config.j
        self.case_dim = config.case_dim
        self.n_cells  = config.max_visible_cells
        stride        = getattr(config, "stride", 20)

        # items : (obs_path, meta_path, agent_id, t)
        self.items: List[Tuple[str, str, int, int]] = []
        self._obs_cache : OrderedDict = OrderedDict()
        self._meta_cache: OrderedDict = OrderedDict()

        for obs_path in sorted(
            glob.glob(os.path.join(obs_dir, "**", "*.npz"), recursive=True)
        ):
            rel       = os.path.relpath(obs_path, obs_dir)
            meta_path = os.path.join(meta_dir, rel)
            if not os.path.exists(meta_path):
                continue

            sess     = np.load(obs_path)
            n_frames, n_agents = sess["actions"].shape
            max_t = n_frames - 1 - self.j
            if max_t < 0:
                continue

            for agent_id in range(n_agents):
                for t in range(0, max_t + 1, stride):
                    self.items.append((obs_path, meta_path, agent_id, t))

    def _load(self, cache: OrderedDict, path: str, maxsize: int = 64):
        if path not in cache:
            if len(cache) >= maxsize:
                cache.popitem(last=False)
            cache[path] = np.load(path)
        return cache[path]

    def __len__(self) -> int:
        return len(self.items)

    # ── Calcul du coût réel ───────────────────────────────────────────────────

    def _compute_cost(self, meta, agent_id: int, t: int) -> float:
        n_meta = len(meta["n_pellets"])
        tj     = min(t + self.j, n_meta - 1)

        if agent_id == 0:
            died = bool(meta["outcome_eaten"]) and (t + self.j >= n_meta - 1)
            if died:
                raw = float(self.j * 10)
            else:
                pellets_eaten = int(meta["n_pellets"][t]) - int(meta["n_pellets"][tj])
                raw = float(self.j - pellets_eaten)
            return (raw - 27.66) / 37.58   # ← normalisation z-score

        else:
            ghost_idx = agent_id - 1
            dist_t  = int(meta["ghost_dist_to_pacman"][t,  ghost_idx])
            dist_tj = int(meta["ghost_dist_to_pacman"][tj, ghost_idx])
            raw = float((2 * dist_tj) - dist_t)
            return (raw - 27.66) / 37.58   # ← même normalisation

    # ── __getitem__ ───────────────────────────────────────────────────────────

    def __getitem__(self, idx):
        obs_path, meta_path, agent_id, t = self.items[idx]

        sess = self._load(self._obs_cache,  obs_path)
        meta = self._load(self._meta_cache, meta_path)

        cases_arr   = sess["cases"][:, agent_id]
        pos_arr     = sess["positions"][:, agent_id]
        masks_arr   = sess["pad_masks"][:, agent_id]
        actions_arr = sess["actions"][:, agent_id]

        # ── Contexte ─────────────────────────────────────────────────────────
        start  = max(0, t - self.seq_len + 1)
        n_real = t + 1 - start
        n_pad  = self.seq_len - n_real

        def pad_left(arr, pad_val=0.0):
            pad_shape = (n_pad,) + arr.shape[1:]
            return np.concatenate(
                [np.full(pad_shape, pad_val, dtype=arr.dtype), arr[start:t+1]], axis=0
            )

        cases_seq  = torch.from_numpy(pad_left(cases_arr))
        pos_seq    = torch.from_numpy(pad_left(pos_arr))
        masks_seq  = torch.from_numpy(pad_left(masks_arr, pad_val=True))

        acts_real    = actions_arr[start:t+1].astype(np.int64)
        acts_real    = np.where(acts_real == -1, 4, acts_real)
        acts_real[0] = 5
        actions_seq  = torch.from_numpy(
            np.concatenate([np.full(n_pad, 5, dtype=np.int64), acts_real])
        )

        # ── actions_future_j (pour compute_gk) ───────────────────────────────
        acts_j           = actions_arr[t:t + self.j].astype(np.int64)
        actions_future_j = torch.from_numpy(np.where(acts_j == -1, 4, acts_j))

        # ── Labels ───────────────────────────────────────────────────────────
        cost_real  = torch.tensor(self._compute_cost(meta, agent_id, t), dtype=torch.float32)
        agent_type = torch.tensor(0 if agent_id == 0 else 1, dtype=torch.long)

        return (
            cases_seq, pos_seq, masks_seq, actions_seq,
            actions_future_j,
            cost_real, agent_type,
        )


# ── Dataset pour le JEPA niveau 1 avec chaîne de 5 pas ───────────────────────

class CostDatasetL1(Dataset):
    """
    Même principe que CostDataset mais pour la chaîne de CHAIN_LEN predictor_1.

    Items retournés :
        cases_seq, pos_seq, masks_seq, actions_seq,   # contexte
        actions_future_5,                             # [CHAIN_LEN] actions réelles
        cost_real,                                    # coût à t+CHAIN_LEN (non normalisé)
        agent_type,                                   # 0=pacman 1=fantôme
    """

    def __init__(self, obs_dir: str, meta_dir: str, config: Config):
        self.seq_len  = config.seq_len
        self.case_dim = config.case_dim
        stride        = getattr(config, "stride", 20)

        self.items: List[Tuple[str, str, int, int]] = []
        self._obs_cache : OrderedDict = OrderedDict()
        self._meta_cache: OrderedDict = OrderedDict()

        for obs_path in sorted(
            glob.glob(os.path.join(obs_dir, "**", "*.npz"), recursive=True)
        ):
            rel       = os.path.relpath(obs_path, obs_dir)
            meta_path = os.path.join(meta_dir, rel)
            if not os.path.exists(meta_path):
                continue

            sess     = np.load(obs_path)
            n_frames, n_agents = sess["actions"].shape
            max_t = n_frames - 1 - CHAIN_LEN
            if max_t < 0:
                continue

            for agent_id in range(n_agents):
                for t in range(0, max_t + 1, stride):
                    self.items.append((obs_path, meta_path, agent_id, t))

    def _load(self, cache: OrderedDict, path: str, maxsize: int = 64):
        if path not in cache:
            if len(cache) >= maxsize:
                cache.popitem(last=False)
            cache[path] = np.load(path)
        return cache[path]

    def __len__(self) -> int:
        return len(self.items)

    def _compute_cost(self, meta, agent_id: int, t: int) -> float:
        """Coût réel à t+CHAIN_LEN, sans normalisation."""
        n_meta = len(meta["n_pellets"])
        t5     = min(t + CHAIN_LEN, n_meta - 1)

        if agent_id == 0:
            died = bool(meta["outcome_eaten"]) and (t + CHAIN_LEN >= n_meta - 1)
            if died:
                return float(CHAIN_LEN * 10)
            pellets_eaten = int(meta["n_pellets"][t]) - int(meta["n_pellets"][t5])
            return float(CHAIN_LEN - pellets_eaten)
        else:
            ghost_idx = agent_id - 1
            dist_t  = int(meta["ghost_dist_to_pacman"][t,  ghost_idx])
            dist_t5 = int(meta["ghost_dist_to_pacman"][t5, ghost_idx])
            return float(2 * dist_t5 - dist_t)

    def __getitem__(self, idx):
        obs_path, meta_path, agent_id, t = self.items[idx]

        sess = self._load(self._obs_cache,  obs_path)
        meta = self._load(self._meta_cache, meta_path)

        cases_arr   = sess["cases"][:, agent_id]
        pos_arr     = sess["positions"][:, agent_id]
        masks_arr   = sess["pad_masks"][:, agent_id]
        actions_arr = sess["actions"][:, agent_id]

        # ── Contexte ─────────────────────────────────────────────────────────
        start  = max(0, t - self.seq_len + 1)
        n_real = t + 1 - start
        n_pad  = self.seq_len - n_real

        def pad_left(arr, pad_val=0.0):
            pad_shape = (n_pad,) + arr.shape[1:]
            return np.concatenate(
                [np.full(pad_shape, pad_val, dtype=arr.dtype), arr[start:t+1]], axis=0
            )

        cases_seq = torch.from_numpy(pad_left(cases_arr))
        pos_seq   = torch.from_numpy(pad_left(pos_arr))
        masks_seq = torch.from_numpy(pad_left(masks_arr, pad_val=True))

        acts_real    = actions_arr[start:t+1].astype(np.int64)
        acts_real    = np.where(acts_real == -1, 4, acts_real)
        acts_real[0] = 5
        actions_seq  = torch.from_numpy(
            np.concatenate([np.full(n_pad, 5, dtype=np.int64), acts_real])
        )

        # ── 5 actions futures ────────────────────────────────────────────────
        acts_5           = actions_arr[t:t + CHAIN_LEN].astype(np.int64)
        actions_future_5 = torch.from_numpy(np.where(acts_5 == -1, 4, acts_5))

        # ── Label ────────────────────────────────────────────────────────────
        cost_real  = torch.tensor(self._compute_cost(meta, agent_id, t), dtype=torch.float32)
        agent_type = torch.tensor(0 if agent_id == 0 else 1, dtype=torch.long)

        return (
            cases_seq, pos_seq, masks_seq, actions_seq,
            actions_future_5,
            cost_real, agent_type,
        )