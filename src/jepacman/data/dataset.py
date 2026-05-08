import os
import glob
import pickle
from typing import List, Tuple
from collections import OrderedDict
import numpy as np

import torch
from torch.utils.data import Dataset

from ..config import Config


def obs_to_tensors(obs: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convertit une observation agent en tenseurs.
    cases     : [N, 8]
    positions : [N, 2]
    pad_mask  : [N]
    """
    cells = obs["visible_cells"]
    cases = torch.tensor([[
            c["wall_north"], c["wall_south"],
            c["wall_east"],  c["wall_west"],
            c["has_pellet"], c["has_super_pellet"],
            c["has_ghost"],  c["has_pacman"],
        ]
        for c in cells
    ], dtype=torch.float32)                              # [N, 8]

    positions = torch.tensor(
        [[c["dx"], c["dy"]] for c in cells],
        dtype=torch.float32
    )                                                    # [N, 2]

    pad_mask = torch.tensor(
        [c["is_pad"] for c in cells],
        dtype=torch.bool
    )                                                    # [N]

    return cases, positions, pad_mask


def sanitize_action(action: int, is_first: bool = False) -> int:
    """
    is_first → 5 (token spécial, pas d'action qui a produit cet état)
    -1       → 4 (stayed)
    0..3     → inchangé
    """
    if is_first:
        return 5
    return 4 if action == -1 else action


def make_padding(n_pad: int, n_cells: int, case_dim: int) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Génère n_pad frames de padding (zéros) à ajouter à gauche de la séquence.
    cases     : [n_pad, N, case_dim]   — zéros
    positions : [n_pad, N, 2]          — zéros
    pad_masks : [n_pad, N]             — True (tout masqué)
    actions   : [n_pad]                — token spécial 5
    """
    cases     = torch.zeros(n_pad, n_cells, case_dim)
    positions = torch.zeros(n_pad, n_cells, 2)
    pad_masks = torch.ones(n_pad, n_cells, dtype=torch.bool)
    actions   = torch.full((n_pad,), 5, dtype=torch.long)
    return cases, positions, pad_masks, actions


class PacmanJEPADataset(Dataset):
    """
    Dataset de séquences anonymes par agent.

    Pour chaque frame t de chaque épisode et chaque agent :
    - Contexte : min(t, seq_len) frames réelles, paddé à gauche jusqu'à seq_len
    - TEM repart toujours de g0 au début du contexte réel (Option A)

    Un item = une séquence d'un seul agent à un instant t.

    Phase 1 retourne :
        cases_seq      [seq_len, N, 8]
        positions_seq  [seq_len, N, 2]
        pad_masks_seq  [seq_len, N]
        actions_seq    [seq_len]
        seq_len_real   int              — nombre de frames réelles (sans padding)
        cases_next     [N, 8]
        positions_next [N, 2]
        pad_mask_next  [N]
        action_next    long

    Phase 2 ajoute :
        cases_tk           [N, 8]
        positions_tk       [N, 2]
        pad_mask_tk        [N]
        actions_future_k   [k]
        cases_tj           [N, 8]
        positions_tj       [N, 2]
        pad_mask_tj        [N]
        actions_future_j   [j]
    """

    def __init__(self, data_dir: str, config: Config):
        self.config    = config
        self.seq_len   = config.seq_len
        self.k         = config.k
        self.j         = config.j
        self.case_dim  = config.case_dim
        self.n_cells   = config.max_visible_cells
        self.phase     = config.phase

        # Chaque item = (agent_obs_list, target_frame_idx)
        # agent_obs_list : liste de toutes les frames de cet agent dans l'épisode
        # target_frame_idx : index de la frame à prédire (t+1)
        self.items : List[Tuple[str, int, int]] = []  # (filepath, agent_id, t)
        self._cache : OrderedDict = OrderedDict()

        for filepath in sorted(glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True)):
            sess = np.load(filepath)
            n_frames, n_agents = sess["actions"].shape
            max_t = n_frames - 1 - self.j

            stride = getattr(config, "stride", 20)
            for agent_id in range(n_agents):
                for t in range(0, max_t + 1, stride):
                    self.items.append((filepath, agent_id, t))

    def _load_session(self, filepath: str):
        if filepath not in self._cache:
            if len(self._cache) >= 64:
                self._cache.popitem(last=False)
            self._cache[filepath] = np.load(filepath)
        return self._cache[filepath]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        filepath, agent_id, t = self.items[idx]
        sess = self._load_session(filepath)

        cases_arr   = sess["cases"][:, agent_id]       # [n_frames, 81, 8]
        pos_arr     = sess["positions"][:, agent_id]   # [n_frames, 81, 2]
        masks_arr   = sess["pad_masks"][:, agent_id]   # [n_frames, 81]
        actions_arr = sess["actions"][:, agent_id]     # [n_frames]

        # ── Contexte ──────────────────────────────────────────────────────────
        start  = max(0, t - self.seq_len + 1)
        n_real = t + 1 - start
        n_pad  = self.seq_len - n_real

        def pad_left(arr, pad_val=0.0):
            pad_shape = (n_pad,) + arr.shape[1:]
            return np.concatenate(
                [np.full(pad_shape, pad_val, dtype=arr.dtype), arr[start:t+1]],
                axis=0
            )

        cases_seq  = torch.from_numpy(pad_left(cases_arr))
        pos_seq    = torch.from_numpy(pad_left(pos_arr))
        masks_seq  = torch.from_numpy(pad_left(masks_arr, pad_val=True))

        actions_real = actions_arr[start:t+1].astype(np.int64)
        actions_real = np.where(actions_real == -1, 4, actions_real)
        actions_real[0] = 5  # is_first token
        actions_seq = torch.from_numpy(
            np.concatenate([np.full(n_pad, 5, dtype=np.int64), actions_real])
        )

        # ── État suivant t+1 ─────────────────────────────────────────────────
        cases_next  = torch.from_numpy(cases_arr[t + 1])
        pos_next    = torch.from_numpy(pos_arr[t + 1])
        pad_next    = torch.from_numpy(masks_arr[t + 1])
        raw_act     = int(actions_arr[t])
        action_next = torch.tensor(4 if raw_act == -1 else raw_act, dtype=torch.long)

        phase1 = (
            cases_seq, pos_seq, masks_seq, actions_seq,
            torch.tensor(n_real, dtype=torch.long),
            cases_next, pos_next, pad_next,
            action_next,
        )

        if self.phase == 1:
            return phase1

        # ── État à t+k ───────────────────────────────────────────────────────
        cases_tk = torch.from_numpy(cases_arr[t + self.k])
        pos_tk   = torch.from_numpy(pos_arr[t + self.k])
        pad_tk   = torch.from_numpy(masks_arr[t + self.k])
        acts_k   = actions_arr[t:t + self.k].astype(np.int64)
        actions_future_k = torch.from_numpy(np.where(acts_k == -1, 4, acts_k))

        # ── État à t+j ───────────────────────────────────────────────────────
        cases_tj = torch.from_numpy(cases_arr[t + self.j])
        pos_tj   = torch.from_numpy(pos_arr[t + self.j])
        pad_tj   = torch.from_numpy(masks_arr[t + self.j])
        acts_j   = actions_arr[t:t + self.j].astype(np.int64)
        actions_future_j = torch.from_numpy(np.where(acts_j == -1, 4, acts_j))

        return (
            *phase1,
            cases_tk, pos_tk, pad_tk, actions_future_k,
            cases_tj, pos_tj, pad_tj, actions_future_j,
        )