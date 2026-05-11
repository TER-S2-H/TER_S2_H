import torch
import torch.nn as nn

class TEMPositionalEncoder(nn.Module):
    def __init__(
        self,
        n_actions: int = 5, # une matrice de transition par action
        d_model: int = 128, # Identique aux modules précédents pour la compatibilité
    ):
        super().__init__()

        self.n_actions = n_actions
        self.d_model = d_model

        # Token spécial pour t=0
        self.g0 = nn.Parameter(torch.randn(1, d_model))

        # Une matrice de transition par action
        # initialisée proche de l'identité pour la stabilité
        eye = torch.eye(d_model).unsqueeze(0).expand(n_actions, -1, -1)
        self.transitions = nn.Parameter(
            eye + 0.01 * torch.randn(n_actions, d_model, d_model)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, actions):
        # actions : [B, T] — action qui a PRODUIT l'état à ce timestep
        #   actions[:, 0] = -1 ou token spécial → pas d'action, on utilise g0
        #   actions[:, 1] = action qui a produit état t=1
        #   actions[:, 2] = action qui a produit état t=2
        #   ...
        actions = actions.clamp(0, len(self.transitions) - 1)
        B, T = actions.shape

        codes = []
        g = self.g0.expand(B, -1)   # [B, d_model] — code initial

        for t in range(T):
            if t == 0:
                # Premier état — pas d'action, on utilise g0 directement
                codes.append(g)
            else:
                # Appliquer la transition de l'action qui a produit cet état
                W_t = self.transitions[actions[:, t]]            # [B, d_model, d_model]
                g   = torch.bmm(g.unsqueeze(1), W_t).squeeze(1) # [B, d_model]
                g   = self.norm(g)
                codes.append(g)

        return torch.stack(codes, dim=1)   # [B, T, d_model]