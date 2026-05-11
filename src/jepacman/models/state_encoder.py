import torch
import torch.nn as nn

class StateEncoder(nn.Module):
    def __init__(
        self,
        case_dim: int = 8,      # Taille du vecteur d'une case
        d_model: int = 128,      # Taille des vecteurs internes du transformer
        n_heads: int = 2,        # Nombre de têtes d'attention
        n_layers: int = 2,       # Nombre de couches de transformer
    ):
        super().__init__()

        self.case_dim = case_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Projette chaque case dans l'espace du transformer
        self.patch_embed = nn.Linear(case_dim, d_model)

        # Encode la position (dx, dy) de chaque case relative à Pacman
        self.pos_embed = nn.Linear(2, d_model)

        # Paramètre appris — vecteur résumé de toute la grille
        # Initialisé aléatoirement, le réseau apprend sa valeur optimale
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # dim_feedforward = d_model * 2 : taille de la couche cachée du FFN
        # batch_first=True : [B, N, d_model] au lieu de [N, B, d_model]
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)

    def forward(self, cases, positions, pad_mask=None):
        B = cases.shape[0]

        x = self.patch_embed(cases) + self.pos_embed(positions)

        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)

        if pad_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)
            pad_mask = torch.cat([cls_mask, pad_mask], dim=1)

        if pad_mask is not None and x.device.type == 'mps':
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            x = self.transformer(x)
        else:
            x = self.transformer(x, src_key_padding_mask=pad_mask)

        return x[:, 0, :]