"""
U-Net architecture and Lightning training module for NewPINNs – 2D SWE
radial dam break.

Clamping is simple: only h > 0 is required for stability.
No EOS / negative-pressure issues.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import lightning.pytorch as pl
from diffusers import UNet2DModel


# ═══════════════════════════════════════════════════════════════════
#  U-Net architecture
# ═══════════════════════════════════════════════════════════════════

class SWEUNet(nn.Module):
    """U-Net  f_NN(t, h_in; θ) → (h, hu, hv) on the spatial grid.

    Input:  (B, 2, ny, nx)  — channel 0 = t, channel 1 = h_in
    Output: (B, 3, ny, nx)  — (h, hu, hv)
    """

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        if config is None:
            config = {}

        sample_size        = config.get("sample_size", (64, 64))
        in_channels        = config.get("in_channels", 2)
        out_channels       = config.get("out_channels", 3)
        layers_per_block   = config.get("layers_per_block", 2)
        block_out_channels = tuple(config.get("block_out_channels", [32, 64, 128]))
        norm_num_groups    = config.get("norm_num_groups", 4)
        down_block_types   = tuple(config.get("down_block_types",
                                ["DownBlock2D"] * len(block_out_channels)))
        up_block_types     = tuple(config.get("up_block_types",
                                ["UpBlock2D"] * len(block_out_channels)))
        attention_head_dim = config.get("attention_head_dim", 4)

        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            norm_num_groups=norm_num_groups,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            attention_head_dim=attention_head_dim,
            act_fn="silu",
        )

        init_method = config.get("init_method", "kaiming")
        if init_method and init_method != "None":
            gain = config.get("init_gain", 0.02)
            self._initialize_weights(init_method, gain)

    def _initialize_weights(self, method: str = "kaiming", gain: float = 0.02) -> None:
        for m in self.unet.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif method == "xavier":
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                elif method == "xavier":
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        print(f"UNet weights initialized: {method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        timesteps = torch.zeros(B, dtype=torch.long, device=x.device)
        return self.unet(x, timesteps).sample


# ═══════════════════════════════════════════════════════════════════
#  Lightning module
# ═══════════════════════════════════════════════════════════════════

class NewPINNsModule(pl.LightningModule):
    """Transient NewPINNs for 2-D shallow water dam break.

    Phase 1: supervised on pretrain set (16 h_in)
    Phase 2: solver-consistency on full training set (256 h_in)
    """

    def __init__(self, net, solver, ic_inputs, ic_target, cfg):
        super().__init__()
        self.net    = net
        self.solver = solver

        self.register_buffer("ic_inputs", ic_inputs)
        self.register_buffer("ic_target", ic_target)

        tc = cfg["training"]
        self.ic_weight         = tc["ic_weight"]
        self.lr                = tc["lr"]
        self.lr_phase2         = tc.get("lr_phase2", self.lr * 0.1)
        self.weight_decay      = tc.get("weight_decay", 1e-4)
        self.max_epochs        = tc["max_epochs"]
        self.pretrain_epochs   = tc.get("pretrain_epochs", 50)
        self.lr_min_factor     = tc.get("lr_min_factor", 0.01)
        self.lr_restart_period = tc.get("lr_restart_period", 100)

        self.criterion = nn.MSELoss()
        self.save_hyperparameters(ignore=["net", "solver", "ic_inputs", "ic_target"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        input_tn, input_tn1, params, target_tn1 = batch

        is_pretrain = self.current_epoch < self.pretrain_epochs

        if is_pretrain:
            pred_tn1 = self.net(input_tn1)
            loss_solver = self.criterion(pred_tn1, target_tn1)
        else:
            pred_tn  = self.net(input_tn)
            pred_tn1 = self.net(input_tn1)

            q_tn = pred_tn.detach().clone()

            # Only constraint: depth h > 0
            q_tn[:, 0].clamp_(min=1e-3)

            solver_out = self.solver.solve(q_tn, params)

            # NaN guard
            bad_mask = torch.isnan(solver_out).flatten(1).any(dim=1) | \
                       torch.isinf(solver_out).flatten(1).any(dim=1)
            n_bad = bad_mask.sum().item()
            if n_bad > 0:
                solver_out[bad_mask] = target_tn1[bad_mask]
                self.log("train/nan_fallbacks", float(n_bad), prog_bar=False)

            loss_solver = self.criterion(pred_tn1, solver_out)

        # IC loss
        ic_pred = self.net(self.ic_inputs)
        loss_ic = self.criterion(ic_pred, self.ic_target)

        loss = loss_solver + self.ic_weight * loss_ic

        self.log("train/loss",        loss,        prog_bar=True)
        self.log("train/solver_loss", loss_solver, prog_bar=False)
        self.log("train/ic_loss",     loss_ic,     prog_bar=False)
        self.log("train/phase",       float(not is_pretrain), prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        preds = self.net(inputs)
        loss  = self.criterion(preds, targets)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.pretrain_epochs:
            self.trainer.datamodule.phase = 2

            optimizer = self.optimizers()
            for pg in optimizer.param_groups:
                pg["lr"] = self.lr_phase2

            scheduler = self.lr_schedulers()
            scheduler.base_lrs = [self.lr_phase2] * len(scheduler.base_lrs)
            scheduler.eta_min  = self.lr_phase2 * self.lr_min_factor
            scheduler.T_0      = self.lr_restart_period
            scheduler.last_epoch = -1

            print(f"\n>>> Phase 2 started (epoch {self.current_epoch}): "
                  f"LR → {self.lr_phase2:.1e}, switched to training set\n")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.lr_restart_period,
            T_mult=1,
            eta_min=self.lr * self.lr_min_factor,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}