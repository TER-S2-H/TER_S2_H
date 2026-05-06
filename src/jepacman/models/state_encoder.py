import torch
import torch.nn as nn

class StateEncoder(nn.Module):
    def __init__(
        self,
        case_dim: int = 9,      # Taille du vecteur d'une case
                                 # contenu (5) + murs (4) = 9
        d_model: int = 128,      # Taille des vecteurs internes du transformer
        n_heads: int = 2,        # Nombre de têtes d'attention
        n_layers: int = 2,       # Nombre de couches de transformer
    ):
        super().__init__()

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

        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, cases, positions):
        # cases     : [B, N, case_dim]
        #   B = batch size
        #   N = nombre de cases = 81 (grille 9×9)
        #   case_dim = 9

        # positions : [B, N, 2]
        #   (dx, dy) de chaque case relative à Pacman

        B = cases.shape[0]

        x = self.patch_embed(cases) + self.pos_embed(positions)

        # Ajouter le CLS token en début de séquence
        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)
        # [B, N+1, d_model]

        x = self.transformer(x)

        # Retourner uniquement le CLS token
        return x[:, 0, :]