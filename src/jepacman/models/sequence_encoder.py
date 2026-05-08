import torch
import torch.nn as nn

class SequenceEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Masque causal — le token t ne voit pas t+1, t+2...
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        self.register_buffer("causal_mask", mask)

    def forward(self, tokens):
        # tokens : [B, T, d_model]
        # state_tokens + Positional encoding déjà additionnés avant d'arriver ici
        x = self.transformer(tokens, mask=self.causal_mask)
        return x   # [B, T, d_model] — on retourne toute la séquence pour le predictor