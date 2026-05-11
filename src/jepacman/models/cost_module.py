"""
Module de coût JEPA-Pac-Man.

Architecture : transformer qui prend en entrée la séquence du sequence encoder
(T tokens) + le token prédit par predictor_3, ce dernier encodé avec g_j comme
positional encoding. Prédit un scalaire (coût futur).

Deux instances sont instanciées : une pour Pac-Man, une pour les fantômes.
"""

import torch
import torch.nn as nn


class CostModule(nn.Module):
    """
    Prédit le coût associé à l'état abstrait futur e_{t+j} prédit par predictor_3.

    Entrées
    -------
    seq    : [B, T, d_model]  — sortie du sequence encoder (contexte courant)
    e_pred : [B, d_model]     — prédiction de predictor_3 (embedding état futur)
    g_j    : [B, d_model]     — variable latente (positional encoding du token futur)

    Sortie
    ------
    [B]  — coût scalaire prédit
    """

    def __init__(
        self,
        d_model:  int = 128,
        n_heads:  int = 2,
        n_layers: int = 2,
    ):
        super().__init__()

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.n_layers = n_layers

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # Tête de régression : d_model → 1
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        seq:    torch.Tensor,  # [B, T, d_model]
        e_pred: torch.Tensor,  # [B, d_model]
        g_j:    torch.Tensor,  # [B, d_model]
    ) -> torch.Tensor:         # [B]
        # Token futur = prédiction + variable latente comme positional encoding
        future_token = (e_pred + g_j).unsqueeze(1)      # [B, 1, d_model]
        tokens = torch.cat([seq, future_token], dim=1)  # [B, T+1, d_model]
        out = self.transformer(tokens)                   # [B, T+1, d_model]
        return self.head(out[:, -1, :]).squeeze(-1)      # [B]