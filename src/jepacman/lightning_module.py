import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from models.jepa import HierarchicalJEPAcman


class HierarchicalJEPAcmanModule(pl.LightningModule):
    def __init__(
        self,
        case_dim: int = 9,
        n_actions: int = 4,
        d_model: int = 128,
        seq_len: int = 128,
        n_heads_state: int = 2,
        n_layers_state: int = 2,
        n_heads_seq: int = 4,
        n_layers_seq: int = 4,
        n_heads_pred: int = 2,
        n_layers_pred: int = 2,
        ema_momentum: float = 0.996,
        k: int = 4,
        j: int = 16,
        lr: float = 3e-4,
        phase: int = 1,   # 1 = encoder + predictor_1, 2 = predictor_2 + predictor_3
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = HierarchicalJEPAcman(
            case_dim=case_dim,
            n_actions=n_actions,
            d_model=d_model,
            seq_len=seq_len,
            n_heads_state=n_heads_state,
            n_layers_state=n_layers_state,
            n_heads_seq=n_heads_seq,
            n_layers_seq=n_layers_seq,
            n_heads_pred=n_heads_pred,
            n_layers_pred=n_layers_pred,
            ema_momentum=ema_momentum,
            k=k,
            j=j,
        )

        self.phase = phase
        self.lr    = lr

    def configure_optimizers(self):
        if self.phase == 1:
            # Phase 1 — encoder + predictor_1 + TEM
            params = (
                list(self.model.state_encoder.parameters())
                + list(self.model.tem.parameters())
                + list(self.model.seq_encoder.parameters())
                + list(self.model.predictor_1.parameters())
            )
        else:
            # Phase 2 — predictor_2 + predictor_3 seulement
            # encoder, TEM, seq_encoder et predictor_1 sont gelés
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
        if self.phase == 1:
            loss = self._step_phase1(batch)
        else:
            loss = self._step_phase2(batch)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.phase == 1:
            loss = self._step_phase1(batch)
        else:
            loss = self._step_phase2(batch)

        self.log("val_loss", loss, prog_bar=True)

    def _step_phase1(self, batch):
        cases, positions, actions, cases_next, positions_next, action_next = batch
        return self.model.forward_level1(
            cases, positions, actions,
            cases_next, positions_next, action_next
        )

    def _step_phase2(self, batch):
        (
            cases, positions, actions,
            cases_tk, positions_tk, actions_future_k,
            cases_tj, positions_tj, actions_future_j,
        ) = batch

        loss_2 = self.model.forward_level2(
            cases, positions, actions,
            cases_tk, positions_tk, actions_future_k
        )
        loss_3 = self.model.forward_level3(
            cases, positions, actions,
            cases_tj, positions_tj, actions_future_j
        )

        loss = loss_2 + loss_3
        self.log("train_loss_level2", loss_2, prog_bar=True)
        self.log("train_loss_level3", loss_3, prog_bar=True)
        return loss

    def on_after_backward(self):
        # EMA uniquement en phase 1 — le target encoder
        # suit l'encoder online
        if self.phase == 1:
            self.model.update_ema()