from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .state_encoder import StateEncoder
from .tem_positional_encoder import TEMPositionalEncoder
from .sequence_encoder import SequenceEncoder
from .predictor import Predictor


class HierarchicalJEPAcman(nn.Module):
    def __init__(
        self,
        case_dim: int = 8,
        n_actions: int = 5,
        d_model: int = 128,
        seq_len: int = 128,
        n_heads_state: int = 2,
        n_layers_state: int = 2,
        n_heads_seq: int = 4,
        n_layers_seq: int = 4,
        n_heads_pred_1: int = 2,
        n_layers_pred_1: int = 2,
        n_heads_pred_2: int = 2,
        n_layers_pred_2: int = 2,
        n_heads_pred_3: int = 2,
        n_layers_pred_3: int = 2,
        ema_momentum: float = 0.996,
        k: int = 4,    # horizon niveau 2
        j: int = 16,   # horizon niveau 3
    ):
        super().__init__()

        self.k = k
        self.j = j
        self.ema_momentum = ema_momentum

        # ── Modules partagés entre tous les niveaux ──────────────────────────

        # TEM partagé — mêmes matrices de transition pour
        # le positional encoding de la séquence ET l'encodage des actions futures
        self.tem = TEMPositionalEncoder(n_actions, d_model)

        # Encoder d'état — ViT sur la grille 9×9
        self.state_encoder = StateEncoder(case_dim, d_model, n_heads_state, n_layers_state)

        # Sequence encoder — transformer temporel causal
        self.seq_encoder = SequenceEncoder(seq_len, d_model, n_heads_seq, n_layers_seq)

        # Target encoder — EMA du state encoder
        self.target_encoder = deepcopy(self.state_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # ── Predicteurs — même architecture, horizons différents ─────────────

        # Niveau 1 — prédit e_{t+1}, conditionné par action (d_model)
        self.predictor_1 = Predictor(n_actions, d_model, n_heads_pred_1, n_layers_pred_1)

        # Niveau 2 — prédit e_{t+k}, conditionné par g_k (d_model)
        # même architecture, le conditionnement est un vecteur d_model dans les deux cas
        self.predictor_2 = Predictor(n_actions, d_model, n_heads_pred_2, n_layers_pred_2)

        # Niveau 3 — prédit e_{t+j}, conditionné par g_j (d_model)
        self.predictor_3 = Predictor(n_actions, d_model, n_heads_pred_3, n_layers_pred_3)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def encode_state(self, cases, positions, online=True):
        # cases     : [B, N, case_dim]
        # positions : [B, N, 2]
        encoder = self.state_encoder if online else self.target_encoder
        return encoder(cases, positions)   # [B, d_model]
    
    def _sanitize_actions(self, actions):
        # Remplacer -1 (stayed) par 4
        return actions.masked_fill(actions == -1, 4)


    def encode_sequence(self, cases, positions, actions, pad_masks=None):
        actions = self._sanitize_actions(actions)
        B, T, N, _ = cases.shape

        masks = pad_masks.view(B * T, N) if pad_masks is not None else None

        state_tokens = self.state_encoder(
            cases.view(B * T, N, -1),
            positions.view(B * T, N, 2),
            pad_mask=masks
        ).view(B, T, -1)

        structural_codes = self.tem(actions)
        tokens = state_tokens + structural_codes
        return self.seq_encoder(tokens)

    def compute_gk(self, actions_future):
        # actions_future : [B, k] — k actions futures réelles
        # retourne uniquement le code final g_k
        codes = self.tem(actions_future)            # [B, k, d_model]
        return codes[:, -1, :]                      # [B, d_model]

    # ── Phase 1 — entraîner encoder + predictor 1 ────────────────────────────

    def forward_level1(self, cases, positions, pad_masks, actions,
                   cases_next, pos_next, pad_next, action_next):
        seq    = self.encode_sequence(cases, positions, actions, pad_masks)
        e_pred = self.predictor_1(seq, action_next)
        with torch.no_grad():
            e_target = self.target_encoder(cases_next, pos_next, pad_mask=pad_next)
        return F.mse_loss(e_pred, e_target)

    def forward_level2(self, cases, positions, pad_masks, actions,
                    cases_tk, pos_tk, pad_tk, actions_future_k):
        with torch.no_grad():
            seq = self.encode_sequence(cases, positions, actions, pad_masks)
            g_k = self.compute_gk(actions_future_k)
        e_pred = self.predictor_2.forward_with_latent(seq, g_k)
        with torch.no_grad():
            e_target = self.target_encoder(cases_tk, pos_tk, pad_mask=pad_tk)
        return F.mse_loss(e_pred, e_target)

    def forward_level3(self, cases, positions, pad_masks, actions,
                    cases_tj, pos_tj, pad_tj, actions_future_j):
        with torch.no_grad():
            seq = self.encode_sequence(cases, positions, actions, pad_masks)
            g_j = self.compute_gk(actions_future_j)
        e_pred = self.predictor_3.forward_with_latent(seq, g_j)
        with torch.no_grad():
            e_target = self.target_encoder(cases_tj, pos_tj, pad_mask=pad_tj)
        return F.mse_loss(e_pred, e_target)

    # ── EMA ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update_ema(self):
        for p_online, p_target in zip(
            self.state_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            p_target.data = (
                self.ema_momentum * p_target.data
                + (1 - self.ema_momentum) * p_online.data
            )

    # ── Inférence hiérarchique ────────────────────────────────────────────────

    @torch.no_grad()
    def _append_prediction(self, seq, e_pred, g_latent):
        # Ajouter une prédiction à la séquence dans la limite du contexte
        # e_pred   : [B, d_model] — état prédit
        # g_latent : [B, d_model] — code structural de ce token
        token = (e_pred + g_latent).unsqueeze(1)   # [B, 1, d_model]
        seq   = torch.cat([seq, token], dim=1)      # [B, T+1, d_model]
        seq   = seq[:, -self.seq_encoder.seq_len:]  # tronquer si dépasse le contexte
        return seq

    def plan_level3(self, seq, cost_fn, n_steps=100, lr=0.01):
        # Optimiser g_j par descente de gradient pour minimiser cost_fn
        # cost_fn : [B, d_model] → [B] — fonction de coût sur un embedding
        B = seq.shape[0]

        g_j = torch.randn(B, seq.shape[-1], requires_grad=True, device=seq.device)
        optimizer = torch.optim.Adam([g_j], lr=lr)

        for _ in range(n_steps):
            optimizer.zero_grad()
            e_goal = self.predictor_3.forward_with_latent(seq.detach(), g_j)
            cost   = cost_fn(e_goal).mean()
            cost.backward()
            optimizer.step()

        return g_j.detach(), self.predictor_3.forward_with_latent(seq.detach(), g_j.detach())
        # g_j : [B, d_model], e_goal : [B, d_model]

    def plan_level2(self, seq, e_goal, n_steps=50, lr=0.01):
        # Prédictions successives de niveau 2 vers e_goal
        # à chaque pas, optimise g_k pour se rapprocher de e_goal
        B, d = seq.shape[0], seq.shape[-1]

        subgoals = []
        current_seq = seq.clone()

        for step in range(self.j // self.k):
            # Optimiser g_k pour ce pas
            g_k = torch.randn(B, d, requires_grad=True, device=seq.device)
            optimizer = torch.optim.Adam([g_k], lr=lr)

            for _ in range(n_steps):
                optimizer.zero_grad()
                e_pred = self.predictor_2.forward_with_latent(current_seq.detach(), g_k)
                loss   = F.mse_loss(e_pred, e_goal)
                loss.backward()
                optimizer.step()

            g_k = g_k.detach()
            with torch.no_grad():
                e_subgoal = self.predictor_2.forward_with_latent(current_seq, g_k)

            subgoals.append(e_subgoal)
            current_seq = self._append_prediction(current_seq, e_subgoal, g_k)

        return subgoals   # liste de j/k embeddings objectifs

    def choose_action(self, seq, e_subgoal):
        # Niveau 1 — choisir l'action qui rapproche le plus du sous-objectif courant
        best_action, best_dist = None, float('inf')

        for a in range(4):
            action = torch.full((seq.shape[0],), a, dtype=torch.long, device=seq.device)
            e_pred = self.predictor_1(seq, action)
            dist   = F.mse_loss(e_pred, e_subgoal).item()
            if dist < best_dist:
                best_dist  = dist
                best_action = a

        return best_action