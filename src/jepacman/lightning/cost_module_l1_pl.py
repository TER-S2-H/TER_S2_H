"""
Module Lightning pour l'entraînement supervisé des modules de coût
avec la chaîne de CHAIN_LEN predictor_1.

Le JEPA est gelé. Seuls cost_pacman et cost_ghost sont entraînés.

Batch attendu (depuis CostDatasetL1) :
    cases_seq, pos_seq, masks_seq, actions_seq,
    actions_future_5,
    cost_real, agent_type
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from ..models.jepa import HierarchicalJEPAcman
from ..models.cost_module import CostModule
from ..data.cost_dataset import CHAIN_LEN


class CostModuleL1Module(pl.LightningModule):

    def __init__(
        self,
        jepa:        HierarchicalJEPAcman,
        cost_pacman: CostModule,
        cost_ghost:  CostModule,
        lr:          float = 3e-4,
    ):
        super().__init__()
        self.jepa        = jepa
        self.cost_pacman = cost_pacman
        self.cost_ghost  = cost_ghost
        self.lr          = lr
        self.save_hyperparameters(ignore=["jepa", "cost_pacman", "cost_ghost"])

    def on_fit_start(self):
        for p in self.jepa.parameters():
            p.requires_grad = False
        self.jepa.eval()

    def on_train_epoch_end(self):
        if self.device.type == "mps":
            torch.mps.empty_cache()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.device.type == "mps" and batch_idx % 100 == 0:
            torch.mps.empty_cache()

    def configure_optimizers(self):
        params = (
            list(self.cost_pacman.parameters())
            + list(self.cost_ghost.parameters())
        )
        return torch.optim.Adam(params, lr=self.lr)

    def _chain_predict(self, seq: torch.Tensor, actions_future_5: torch.Tensor):
        """
        Chaîne CHAIN_LEN appels à predictor_1.
        seq               : [B, T, d_model]
        actions_future_5  : [B, CHAIN_LEN]
        Retourne e_final [B, d_model] et g_chain [B, d_model].
        """
        seq_len     = self.jepa.seq_encoder.seq_len
        current_seq = seq
        e_step      = None

        for t in range(CHAIN_LEN):
            acts_t = actions_future_5[:, t]                       # [B]
            g_a    = self.jepa.predictor_1.action_embed(acts_t)  # [B, d_model]
            e_step = self.jepa.predictor_1(current_seq, acts_t)  # [B, d_model]
            token  = (e_step + g_a).unsqueeze(1)                 # [B, 1, d_model]
            current_seq = torch.cat([current_seq, token], dim=1)
            current_seq = current_seq[:, -seq_len:]

        g_chain = self.jepa.tem(actions_future_5)[:, -1, :]      # [B, d_model]
        return e_step, g_chain

    def _step(self, batch):
        (
            cases_seq, pos_seq, masks_seq, actions_seq,
            actions_future_5,
            cost_real, agent_type,
        ) = batch

        with torch.no_grad():
            seq = self.jepa.encode_sequence(
                cases_seq, pos_seq, actions_seq, masks_seq
            )
            e_final, g_chain = self._chain_predict(seq, actions_future_5)

        pac_mask   = agent_type == 0
        ghost_mask = agent_type == 1

        loss = torch.tensor(0.0, device=self.device)

        if pac_mask.any():
            c_pred   = self.cost_pacman(
                seq[pac_mask], e_final[pac_mask], g_chain[pac_mask]
            )
            loss_pac = F.huber_loss(c_pred, cost_real[pac_mask], delta=1.0)
            self.log("pac_loss",  loss_pac,  prog_bar=True, on_step=False, on_epoch=True)
            loss = loss + loss_pac

        if ghost_mask.any():
            c_pred     = self.cost_ghost(
                seq[ghost_mask], e_final[ghost_mask], g_chain[ghost_mask]
            )
            loss_ghost = F.huber_loss(c_pred, cost_real[ghost_mask], delta=1.0)
            self.log("ghost_loss", loss_ghost, prog_bar=True, on_step=False, on_epoch=True)
            loss = loss + loss_ghost

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
