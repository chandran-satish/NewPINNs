"""
Neural-network architecture and Lightning training module for NewPINNs.
Adapted for the 1-D variable-velocity advection equation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import lightning.pytorch as pl


# ═══════════════════════════════════════════════════════════════════
#  MLP architecture
# ═══════════════════════════════════════════════════════════════════

_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


class Net(nn.Module):
    """Fully-connected network  f_NN(t, α, x; θ) → q̂."""

    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 5,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        act_cls = _ACTIVATIONS[activation]

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act_cls())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_cls())
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════
#  PyTorch Lightning module
# ═══════════════════════════════════════════════════════════════════

class NewPINNsModule(pl.LightningModule):
    """Transient NewPINNs training loop (Algorithm 1 from the paper).

    Each training_step receives a batch of *paired* inputs
    ``(inputs_tn, inputs_tn1)`` — one for time-step n and one for n+1.

    1. Network predicts q̂ at both time levels.
    2. Solver advances q̂(t_n) → q_solver(t_{n+1}).
    3. L_solver = MSE(q̂(t_{n+1}),  q_solver(t_{n+1}))
    4. L_IC     = MSE(q̂(t_0),  q_0)
    5. L_total  = L_solver + w · L_IC
    """

    def __init__(self, net, solver, first_time, ic_target, cfg):
        super().__init__()
        self.net    = net
        self.solver = solver

        # Buffers automatically move to the correct device.
        self.register_buffer("first_time", first_time)   # (R·N, 3)
        self.register_buffer("ic_target",  ic_target)    # (R·N, 1)

        tc = cfg["training"]
        self.ic_weight        = tc["ic_weight"]
        self.lr               = tc["lr"]
        self.weight_decay     = tc.get("weight_decay", 1e-4)
        self.max_epochs       = tc["max_epochs"]
        self.lr_min_factor    = tc.get("lr_min_factor", 0.01)
        self.lr_restart_period = tc.get("lr_restart_period", 50)

        self.criterion = nn.MSELoss()
        self.save_hyperparameters(ignore=["net", "solver", "first_time", "ic_target"])

    # ── forward ────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ── training step ──────────────────────────────────────────────

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # batch shape: (B, 2, N_spatial, 3)
        B = batch.shape[0]
        N = batch.shape[2]

        # Network predictions at t_n and t_{n+1}
        flat     = batch.reshape(-1, 3)           # (2·B·N, 3)
        pred     = self.net(flat).view(2 * B, N, -1)
        pred_tn  = pred[0::2]                     # (B, N, 1)
        pred_tn1 = pred[1::2]                     # (B, N, 1)

        # α is constant over the spatial dimension and identical at both times
        alpha = batch[:, 0, 0, 1]                 # (B,)

        # Solver: advance q̂(t_n) → q_solver(t_{n+1}).
        # Clamp to a physically reasonable range so wild early-training
        # predictions don't destabilise the advection solver.
        # The IC is O(1) so [-3, 3] is a generous but safe bound.
        q_tn = pred_tn.squeeze(-1).detach().clamp(-3.0, 3.0)  # (B, N)
        solver_out = self.solver.solve(q_tn, alpha.detach())   # (B, N)

        loss_solver = self.criterion(pred_tn1.squeeze(-1), solver_out)

        # IC loss
        ic_pred = self.net(self.first_time)        # (R·N, 1)
        loss_ic = self.criterion(ic_pred, self.ic_target)

        loss = loss_solver + self.ic_weight * loss_ic

        self.log("train/loss",        loss,         prog_bar=True)
        self.log("train/solver_loss", loss_solver,  prog_bar=False)
        self.log("train/ic_loss",     loss_ic,      prog_bar=False)
        return loss

    # ── validation step ────────────────────────────────────────────

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        inputs, targets = batch          # (B, 3), (B, 1)
        preds = self.net(inputs)         # (B, 1)
        loss  = self.criterion(preds, targets)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    # ── optimiser ──────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.lr_restart_period,   # epochs per restart cycle
            T_mult=1,                     # cycle length stays fixed
            eta_min=self.lr * self.lr_min_factor,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}