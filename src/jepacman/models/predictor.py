import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(
        self,
        n_actions: int = 5,
        d_model: int = 128,
        n_heads: int = 2,
        n_layers: int = 2,
    ):
        super().__init__()

        self.n_actions = n_actions
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Embedding de l'action candidate
        self.action_embed = nn.Embedding(n_actions, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, seq, action):
        # seq : [B, T, d_model] — output du sequence encoder
        # action : [B] — action candidate 0..3

        # Conditionner toute la séquence avec l'action candidate
        a = self.action_embed(action).unsqueeze(1)       # [B, 1, d_model]
        tokens = seq + a                                 # [B, T, d_model]

        x = self.transformer(tokens)                     # [B, T, d_model]
        return x[:, -1, :]                               # [B, d_model]
    
    def forward_with_latent(self, seq, latent):
        # latent : [B, d_model] — vecteur continu
        # utilisé pour les niveaux 2 et 3
        tokens = seq + latent.unsqueeze(1)   # [B, T, d_model]
        x = self.transformer(tokens)
        return x[:, -1, :]