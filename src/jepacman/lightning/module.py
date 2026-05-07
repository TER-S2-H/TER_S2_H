import torch
import pytorch_lightning as pl

from config import Config
from models.jepa import HierarchicalJEPAcman


class HierarchicalJEPAcmanModule(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.phase  = config.phase
        self.lr     = config.lr

        # save_hyperparameters sur le dataclass converti en dict
        self.save_hyperparameters(config.__dict__)

        self.model = HierarchicalJEPAcman(
            case_dim       = config.case_dim,
            n_actions      = config.n_actions,
            d_model        = config.d_model,
            seq_len        = config.seq_len,
            n_heads_state  = config.n_heads_state,
            n_layers_state = config.n_layers_state,
            n_heads_seq    = config.n_heads_seq,
            n_layers_seq   = config.n_layers_seq,
            n_heads_pred_1 = config.n_heads_pred_1,
            n_layers_pred_1= config.n_layers_pred_1,
            n_heads_pred_2 = config.n_heads_pred_2,
            n_layers_pred_2= config.n_layers_pred_2,
            n_heads_pred_3 = config.n_heads_pred_3,
            n_layers_pred_3= config.n_layers_pred_3,
            ema_momentum   = config.ema_momentum,
            k              = config.k,
            j              = config.j,
        )

    def configure_optimizers(self):
        if self.phase == 1:
            params = (
                list(self.model.state_encoder.parameters())
                + list(self.model.tem.parameters())
                + list(self.model.seq_encoder.parameters())
                + list(self.model.predictor_1.parameters())
            )
        else:
            params = (
                list(self.model.predictor_2.parameters())
                + list(self.model.predictor_3.parameters())
            )
        return torch.optim.Adam(params, lr=self.lr)

    def _freeze_encoder(self):
        for module in [
            self.model.state_encoder,
            self.model.tem,
            self.model.seq_encoder,
            self.model.predictor_1,
        ]:
            for p in module.parameters():
                p.requires_grad = False

    def on_fit_start(self):
        if self.phase == 2:
            self._freeze_encoder()

    def training_step(self, batch, batch_idx):
        loss = self._step_phase1(batch) if self.phase == 1 else self._step_phase2(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step_phase1(batch) if self.phase == 1 else self._step_phase2(batch)
        self.log("val_loss", loss, prog_bar=True)

    def _unpack_context(self, batch):
        """Extrait les éléments communs aux deux phases"""
        (
            cases_seq, pos_seq, pad_masks_seq, actions_seq, n_real,
            cases_next, pos_next, pad_next, action_next,
            *rest
        ) = batch
        return (
            cases_seq, pos_seq, pad_masks_seq, actions_seq, n_real,
            cases_next, pos_next, pad_next, action_next,
            rest
        )

    def _step_phase1(self, batch):
        (
            cases_seq, pos_seq, pad_masks_seq, actions_seq, n_real,
            cases_next, pos_next, pad_next, action_next,
            _
        ) = self._unpack_context(batch)

        return self.model.forward_level1(
            cases_seq, pos_seq, pad_masks_seq, actions_seq,
            cases_next, pos_next, pad_next,
            action_next,
        )

    def _step_phase2(self, batch):
        (
            cases_seq, pos_seq, pad_masks_seq, actions_seq, n_real,
            cases_next, pos_next, pad_next, action_next,
            rest
        ) = self._unpack_context(batch)

        (
            cases_tk, pos_tk, pad_tk, actions_future_k,
            cases_tj, pos_tj, pad_tj, actions_future_j,
        ) = rest

        loss_2 = self.model.forward_level2(
            cases_seq, pos_seq, pad_masks_seq, actions_seq,
            cases_tk, pos_tk, pad_tk,
            actions_future_k,
        )
        loss_3 = self.model.forward_level3(
            cases_seq, pos_seq, pad_masks_seq, actions_seq,
            cases_tj, pos_tj, pad_tj,
            actions_future_j,
        )

        loss = loss_2 + loss_3
        self.log("train_loss_level2", loss_2, prog_bar=True)
        self.log("train_loss_level3", loss_3, prog_bar=True)
        return loss

    def on_after_backward(self):
        if self.phase == 1:
            self.model.update_ema()