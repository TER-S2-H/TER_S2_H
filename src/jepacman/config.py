from dataclasses import dataclass

@dataclass
class Config:

    # ── Données ───────────────────────────────────────────────────────────────
    max_visible_cells : int   = 81    # taille fixe après padding (9×9)
    case_dim          : int   = 8     # 4 murs + 4 contenus (dx/dy sont des positions)
    n_actions         : int   = 5     # 0=haut 1=bas 2=gauche 3=droite 4=stayed
    seq_len           : int   = 128   # nombre de timesteps dans le contexte

    # ── Horizons hiérarchiques ────────────────────────────────────────────────
    k : int = 4    # horizon niveau 2 — prédit e_{t+k}
    j : int = 16   # horizon niveau 3 — prédit e_{t+j}
                   # j doit être divisible par k

    # ── Architecture ─────────────────────────────────────────────────────────
    d_model : int = 128   # taille des embeddings — partagée entre tous les modules

    # State encoder (ViT)
    n_heads_state  : int = 2
    n_layers_state : int = 2

    # Sequence encoder
    n_heads_seq    : int = 4
    n_layers_seq   : int = 4

    # Predictor — même architecture pour les 3 niveaux
    n_heads_pred_1   : int = 2
    n_layers_pred_1  : int = 2

    n_heads_pred_2   : int = 2
    n_layers_pred_2  : int = 2

    n_heads_pred_3   : int = 2
    n_layers_pred_3  : int = 2

    # ── JEPA ─────────────────────────────────────────────────────────────────
    ema_momentum : float = 0.996

    # ── Entraînement ─────────────────────────────────────────────────────────
    lr            : float = 3e-4
    batch_size    : int   = 64
    phase         : int   = 1     # 1 = encoder + predictor_1 / 2 = predictor_2 + predictor_3
    stride : int = 20   # ne prendre qu'une frame sur 20

    # ── Inférence ─────────────────────────────────────────────────────────────
    latent_optim_steps : int   = 100    # steps de descente de gradient sur g_j / g_k
    latent_optim_lr    : float = 0.01

    def __post_init__(self):
        assert self.j % self.k == 0, "j doit être divisible par k"
        assert self.d_model % self.n_heads_state == 0, "d_model doit être divisible par n_heads_state"
        assert self.d_model % self.n_heads_seq   == 0, "d_model doit être divisible par n_heads_seq"
        assert self.d_model % self.n_heads_pred_1  == 0, "d_model doit être divisible par n_heads_pred_1"
        assert self.d_model % self.n_heads_pred_2  == 0, "d_model doit être divisible par n_heads_pred_2"
        assert self.d_model % self.n_heads_pred_3  == 0, "d_model doit être divisible par n_heads_pred_3"