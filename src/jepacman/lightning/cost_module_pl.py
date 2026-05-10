import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from ..models.jepa import HierarchicalJEPAcman
from ..models.cost_module import CostModule


class CostModuleModule(pl.LightningModule):
    """
    Module Lightning pour l'entraînement supervisé des modules de coût.

    Le modèle JEPA est gelé. Seuls cost_pacman et cost_ghost sont entraînés.

    Batch attendu (depuis CostDataset) :
        cases_seq, pos_seq, masks_seq, actions_seq,
        actions_future_j,
        cost_real, agent_type
    """

    def __init__(
        self,
        jepa:         HierarchicalJEPAcman,
        cost_pacman:  CostModule,
        cost_ghost:   CostModule,
        lr:           float = 3e-4,
    ):
        super().__init__()
        self.jepa        = jepa
        self.cost_pacman = cost_pacman
        self.cost_ghost  = cost_ghost
        self.lr          = lr
        self.save_hyperparameters(ignore=["jepa", "cost_pacman", "cost_ghost"])

    def on_fit_start(self):
        # Geler JEPA entièrement
        for p in self.jepa.parameters():
            p.requires_grad = False
        self.jepa.eval()

    def on_train_epoch_end(self):
        if self.device.type == 'mps':
            torch.mps.empty_cache()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.device.type == 'mps' and batch_idx % 100 == 0:
            torch.mps.empty_cache()

    def configure_optimizers(self):
        params = (
            list(self.cost_pacman.parameters())
            + list(self.cost_ghost.parameters())
        )
        return torch.optim.Adam(params, lr=self.lr)

    def _step(self, batch):
        (
            cases_seq, pos_seq, masks_seq, actions_seq,
            actions_future_j,
            cost_real, agent_type,
        ) = batch

        with torch.no_grad():
            seq    = self.jepa.encode_sequence(
                cases_seq, pos_seq, actions_seq, masks_seq
            )
            g_j    = self.jepa.compute_gk(actions_future_j)
            e_pred = self.jepa.predictor_3.forward_with_latent(seq, g_j)

        pac_mask   = agent_type == 0
        ghost_mask = agent_type == 1

        loss = torch.tensor(0.0, device=self.device)

        if pac_mask.any():
            c_pred = self.cost_pacman(
                seq[pac_mask], e_pred[pac_mask], g_j[pac_mask]
            )
            loss_pac = F.mse_loss(c_pred, cost_real[pac_mask])
            self.log("pac_loss", loss_pac, prog_bar=True,
                     on_step=False, on_epoch=True)
            loss = loss + loss_pac

        if ghost_mask.any():
            c_pred = self.cost_ghost(
                seq[ghost_mask], e_pred[ghost_mask], g_j[ghost_mask]
            )
            loss_ghost = F.mse_loss(c_pred, cost_real[ghost_mask])
            self.log("ghost_loss", loss_ghost, prog_bar=True,
                     on_step=False, on_epoch=True)
            loss = loss + loss_ghost

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)