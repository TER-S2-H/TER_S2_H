import os
import pickle
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from config import Config


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
        self.items : List[Tuple[List[dict], int]] = []

        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".pkl"):
                continue

            with open(os.path.join(data_dir, fname), "rb") as f:
                session = pickle.load(f)

            n_frames  = len(session)
            n_agents  = len(session[0])

            # Extraire les observations par agent
            for agent_id in range(n_agents):
                agent_obs = [session[t][agent_id] for t in range(n_frames)]

                # Pour chaque frame t, le contexte va de max(0, t-seq_len) à t
                # et l'état à prédire est t+1 (donc t doit aller jusqu'à n_frames - 1 - j)
                min_t = 0 # dès la première frame
                max_t = n_frames - 1 - self.j  # assez de futur pour t+j

                for t in range(min_t, max_t + 1):
                    self.items.append((agent_obs, t))

    def __len__(self):
        return len(self.items)

    def _build_context(self, agent_obs: List[dict], t: int):
        """
        Construit la séquence de contexte de longueur seq_len pour la frame t.
        Les frames réelles vont de max(0, t - seq_len + 1) à t inclus.
        Le reste est paddé à gauche avec des zéros.
        """
        start      = max(0, t - self.seq_len + 1)
        real_frames = agent_obs[start : t + 1]          # liste de dicts
        n_real     = len(real_frames)                    # <= seq_len
        n_pad      = self.seq_len - n_real

        # Convertir les frames réelles
        all_cases, all_positions, all_pad_masks, all_actions = [], [], [], []

        for i, obs in enumerate(real_frames):
            cases, pos, pad = obs_to_tensors(obs)
            all_cases.append(cases)
            all_positions.append(pos)
            all_pad_masks.append(pad)
            # is_first=True uniquement pour la toute première frame réelle
            all_actions.append(sanitize_action(obs["action"], is_first=(i == 0)))

        cases_real    = torch.stack(all_cases)           # [n_real, N, 8]
        pos_real      = torch.stack(all_positions)       # [n_real, N, 2]
        masks_real    = torch.stack(all_pad_masks)       # [n_real, N]
        actions_real  = torch.tensor(all_actions, dtype=torch.long)  # [n_real]

        # Ajouter le padding à gauche si nécessaire
        if n_pad > 0:
            pad_c, pad_p, pad_m, pad_a = make_padding(
                n_pad, self.n_cells, self.case_dim
            )
            cases_seq   = torch.cat([pad_c, cases_real],   dim=0)  # [seq_len, N, 8]
            pos_seq     = torch.cat([pad_p, pos_real],     dim=0)  # [seq_len, N, 2]
            masks_seq   = torch.cat([pad_m, masks_real],   dim=0)  # [seq_len, N]
            actions_seq = torch.cat([pad_a, actions_real], dim=0)  # [seq_len]
        else:
            cases_seq   = cases_real
            pos_seq     = pos_real
            masks_seq   = masks_real
            actions_seq = actions_real

        return cases_seq, pos_seq, masks_seq, actions_seq, n_real

    def __getitem__(self, idx):
        agent_obs, t = self.items[idx]

        # ── Contexte ──────────────────────────────────────────────────────────
        cases_seq, pos_seq, masks_seq, actions_seq, n_real = (
            self._build_context(agent_obs, t)
        )

        # ── État suivant t+1 ─────────────────────────────────────────────────
        obs_next               = agent_obs[t + 1]
        cases_next, pos_next, pad_next = obs_to_tensors(obs_next)
        action_next            = torch.tensor(
            sanitize_action(agent_obs[t]["action"]),
            dtype=torch.long
        )

        phase1 = (
            cases_seq, pos_seq, masks_seq, actions_seq,
            torch.tensor(n_real, dtype=torch.long),
            cases_next, pos_next, pad_next,
            action_next,
        )

        if self.phase == 1:
            return phase1

        # ── État à t+k ───────────────────────────────────────────────────────
        obs_tk                     = agent_obs[t + self.k]
        cases_tk, pos_tk, pad_tk   = obs_to_tensors(obs_tk)
        actions_future_k           = torch.tensor([
            sanitize_action(agent_obs[t + i]["action"])
            for i in range(self.k)
        ], dtype=torch.long)                                   # [k]

        # ── État à t+j ───────────────────────────────────────────────────────
        obs_tj                     = agent_obs[t + self.j]
        cases_tj, pos_tj, pad_tj   = obs_to_tensors(obs_tj)
        actions_future_j           = torch.tensor([
            sanitize_action(agent_obs[t + i]["action"])
            for i in range(self.j)
        ], dtype=torch.long)                                   # [j]

        return (
            *phase1,
            cases_tk, pos_tk, pad_tk, actions_future_k,
            cases_tj, pos_tj, pad_tj, actions_future_j,
        )