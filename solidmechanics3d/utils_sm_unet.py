import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils_sm_ngsolve import Elasticity3DSolver
from utils_sm_data import (
    E_FIXED, FORCE_VEC, NU_MIN, NU_MAX,
    MAXH, FE_ORDER, N_HOLES, CHAMFER, BOX_DIMS,
    NZ, NY, NX, TAU,
)


torch.set_float32_matmul_precision("medium")


# ======================================================================
# 3D U-Net (custom, since diffusers has no plain UNet3DModel)
# ======================================================================

class DoubleConv3D(nn.Module):
    """Conv → GroupNorm → SiLU, twice, with a residual projection."""

    def __init__(self, in_ch, out_ch, norm_num_groups=8):
        super().__init__()
        groups1 = min(norm_num_groups, in_ch)
        groups2 = min(norm_num_groups, out_ch)
        self.norm1 = nn.GroupNorm(groups1, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups2, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.proj = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.proj(x)


class UNet3D(nn.Module):
    """
    Compact 3D U-Net with `len(block_out_channels) - 1` downsampling stages
    and a matching number of upsampling stages. Handles non-cubic and
    non-divisible-by-2 spatial dimensions via size-matching trilinear
    interpolation on the decoder skip path.

    Args:
        in_channels: number of input channels (here: 2 = ν_norm + mask)
        out_channels: number of output channels (here: 3 = U, V, W)
        block_out_channels: per-level channel widths from shallowest to
            deepest. The last entry is the bottleneck width.
        norm_num_groups: GroupNorm group count (clamped to channel count).
    """

    def __init__(
        self,
        in_channels=2,
        out_channels=3,
        block_out_channels=(64, 128, 256),
        norm_num_groups=8,
    ):
        super().__init__()
        self.block_out_channels = tuple(block_out_channels)

        # Encoder: a DoubleConv at each level + a strided conv to halve resolution.
        self.enc_convs = nn.ModuleList()
        self.enc_downs = nn.ModuleList()
        prev_ch = in_channels
        for ch in block_out_channels[:-1]:
            self.enc_convs.append(DoubleConv3D(prev_ch, ch, norm_num_groups))
            self.enc_downs.append(
                nn.Conv3d(ch, ch, kernel_size=3, stride=2, padding=1)
            )
            prev_ch = ch

        # Bottleneck (no downsample).
        self.bottleneck = DoubleConv3D(
            prev_ch, block_out_channels[-1], norm_num_groups
        )

        # Decoder: ConvTranspose to double resolution, concat skip, DoubleConv.
        self.dec_ups = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        prev_ch = block_out_channels[-1]
        for ch in reversed(block_out_channels[:-1]):
            self.dec_ups.append(
                nn.ConvTranspose3d(prev_ch, ch, kernel_size=4,
                                   stride=2, padding=1)
            )
            self.dec_convs.append(
                DoubleConv3D(ch + ch, ch, norm_num_groups)
            )
            prev_ch = ch

        # 1x1x1 conv to displacement channels.
        self.out_conv = nn.Conv3d(block_out_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for conv, down in zip(self.enc_convs, self.enc_downs):
            x = conv(x)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.dec_ups, self.dec_convs, reversed(skips)):
            x = up(x)
            # Reconcile size if a dimension was odd at some encoder level.
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:],
                    mode="trilinear", align_corners=False,
                )
            x = torch.cat([x, skip], dim=1)
            x = conv(x)
        return self.out_conv(x)


# ======================================================================
# Lightning wrapper (mirrors cavity ImagePredictorUNet)
# ======================================================================

class ImagePredictorUNet3D(nn.Module):
    """
    Wraps UNet3D with the same config dict / output-range bookkeeping
    pattern used in the cavity model. Output is the raw UNet prediction;
    range knobs in config are informational defaults (no scaling applied
    in forward, matching the cavity).
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}

        in_channels       = config.get("in_channels", 2)
        out_channels      = config.get("out_channels", 3)
        block_out_channels = tuple(config.get("block_out_channels", (64, 128, 256)))
        norm_num_groups   = config.get("norm_num_groups", 8)

        self.unet = UNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            norm_num_groups=norm_num_groups,
        )

        # Informational output ranges (not enforced in forward).
        self.U_min = config.get("U_min", -0.6)
        self.U_max = config.get("U_max",  0.6)
        self.V_min = config.get("V_min", -0.1)
        self.V_max = config.get("V_max",  0.1)
        self.W_min = config.get("W_min", -0.1)
        self.W_max = config.get("W_max",  0.1)

        # Weight init (same options as cavity model).
        init_method = config.get("init_method", "kaiming")
        if init_method == "None":
            init_method = None
        gain = config.get("init_gain", 0.02)
        if init_method is not None:
            self.initialize_weights(init_method, gain)

    def initialize_weights(self, method="kaiming", gain=0.02):
        for m in self.unet.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                if method == "kaiming":
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif method == "xavier":
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                elif method == "orthogonal":
                    nn.init.orthogonal_(m.weight, gain=gain)
                elif method == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=gain)
                elif method == "zeros":
                    nn.init.zeros_(m.weight)
                elif method == "near_zero":
                    nn.init.normal_(m.weight, mean=0.0, std=gain / 10.0)
                if m.bias is not None:
                    if method == "near_zero":
                        nn.init.normal_(m.bias, mean=0.0, std=gain / 10.0)
                    else:
                        nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.GroupNorm):
                if method == "near_zero":
                    nn.init.normal_(m.weight, mean=1.0, std=gain / 10.0)
                    nn.init.normal_(m.bias,   mean=0.0, std=gain / 10.0)
                else:
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias,   0.0)
        print(f"UNet3D weights initialized using {method}"
              + (f" with gain={gain}" if method in
                 ("xavier", "orthogonal", "normal", "near_zero") else ""))

    def forward(self, x):
        # x has shape (B, in_channels, nz, ny, nx); pass straight through.
        return self.unet(x)


# ======================================================================
# Mask-aware MSE
# ======================================================================

def masked_mse(pred, target, mask, eps=1e-8):
    """
    Mean-squared error averaged only over entries where mask > 0.5.

    pred, target: (B, C, ...). mask: (B, 1, ...) or broadcastable.
    """
    sq = (pred - target) ** 2
    m = mask.to(pred.dtype)
    num = (sq * m).sum()
    den = (m.expand_as(sq).sum()).clamp(min=eps)
    return num / den


# ======================================================================
# Lightning module (mirrors cavity FEMPhysicsModule)
# ======================================================================

class FEMPhysicsModule(pl.LightningModule):
    def __init__(
        self,
        model_type="unet",
        learning_rate=1e-4,
        fem_iterations=20,
        Tmax=100,
        lambda_u=1.0,
        lambda_v=1.0,
        lambda_w=1.0,
        model_config=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if model_type == "unet":
            self.model = ImagePredictorUNet3D(model_config)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.criterion = masked_mse
        self.learning_rate = learning_rate
        self.fem_iterations = fem_iterations
        self.Tmax = Tmax
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_w = lambda_w

        self.train_losses = []
        self.val_losses = []

        # Per-rank persistent solver, created lazily.
        self._solver = None

        # Per-epoch nan/inf bookkeeping.
        self.nan_inf_indices = []
        self.epoch_nan_inf_count = 0
        self.total_nan_inf_count = 0

    # ------------------------------------------------------------------
    def _get_solver(self):
        if self._solver is None:
            # in utils_sm_unet.py around line 274
            self._solver = Elasticity3DSolver(
                maxh=MAXH, order=FE_ORDER, n_holes=N_HOLES,
                box_dims=BOX_DIMS, chamfer=CHAMFER,
                nx=NX, ny=NY, nz=NZ,        # drop tau=TAU
            )
            print(f"[Rank {self.global_rank}] Created persistent "
                  f"Elasticity3DSolver", flush=True)
        return self._solver

    # ------------------------------------------------------------------
    def forward(self, x):
        return self.model(x)

    # ------------------------------------------------------------------
    def _process_train_batch(self, batch, batch_idx):
        """
        NewPINNs solver-consistency step.

          1. Network output := f_NN(ν_norm, mask).
          2. Solver iterates the network output for fem_iterations
             Richardson steps with BDDC, returning the solver-advanced
             displacement field.
          3. Loss := masked MSE between (network output) and (solver
             output), summed across the three displacement channels.

        This is the steady-state NewPINNs objective (Eq. 17, Prop. 1).
        """
        img1 = batch  # (B, 2, nz, ny, nx): channels [ν_norm, mask]
        output = self(img1)  # (B, 3, nz, ny, nx)

        # Mask is channel 1 of the input.
        mask_b = img1[:, 1:2]  # keep dim → (B, 1, nz, ny, nx)

        batch_t_u, batch_t_v, batch_t_w = [], [], []
        valid_indices = []
        batch_size = output.size(0)
        global_start_idx = batch_idx * batch_size

        solver = self._get_solver()

        for i in range(batch_size):
            nu_norm_arr = img1[i, 0].detach().cpu().numpy()
            single_output = output[i].detach().cpu().numpy()  # (3, nz, ny, nx)

            nu_poisson = (NU_MAX - NU_MIN) * float(nu_norm_arr[0, 0, 0]) + NU_MIN

            t_u, t_v, t_w, _ = solver.solve(
                E=E_FIXED,
                nu_poisson=nu_poisson,
                force_vec=FORCE_VEC,
                n_iter=self.fem_iterations,
                U_initial=single_output[0],
                V_initial=single_output[1],
                W_initial=single_output[2],
            )

            arrs = (t_u, t_v, t_w)
            if any(np.isnan(a).any() or np.isinf(a).any() for a in arrs):
                global_idx = global_start_idx + i
                self.nan_inf_indices.append(global_idx)
                self.epoch_nan_inf_count += 1
                self.total_nan_inf_count += 1
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{ts}] Error: NaN/Inf detected in sample {global_idx}",
                      flush=True)
                continue

            t_u = torch.from_numpy(t_u).float().unsqueeze(0).unsqueeze(0).to(self.device)
            t_v = torch.from_numpy(t_v).float().unsqueeze(0).unsqueeze(0).to(self.device)
            t_w = torch.from_numpy(t_w).float().unsqueeze(0).unsqueeze(0).to(self.device)
            batch_t_u.append(t_u)
            batch_t_v.append(t_v)
            batch_t_w.append(t_w)
            valid_indices.append(i)

        if not valid_indices:
            return None, 0

        target = torch.cat([
            torch.cat(batch_t_u, dim=0),
            torch.cat(batch_t_v, dim=0),
            torch.cat(batch_t_w, dim=0),
        ], dim=1)  # (B_valid, 3, nz, ny, nx)

        valid_output = output[valid_indices]
        valid_mask = mask_b[valid_indices]

        pred_u, pred_v, pred_w = valid_output[:, 0:1], valid_output[:, 1:2], valid_output[:, 2:3]
        true_u, true_v, true_w = target[:, 0:1],       target[:, 1:2],       target[:, 2:3]

        loss_u = self.criterion(pred_u, true_u, valid_mask)
        loss_v = self.criterion(pred_v, true_v, valid_mask)
        loss_w = self.criterion(pred_w, true_w, valid_mask)
        loss = (self.lambda_u * loss_u
                + self.lambda_v * loss_v
                + self.lambda_w * loss_w)
        return loss, len(valid_indices)

    # ------------------------------------------------------------------
    def _process_val_batch(self, batch, batch_idx):
        img1, Ueq, Veq, Weq = batch
        output = self(img1)  # (B, 3, nz, ny, nx)
        mask_b = img1[:, 1:2]

        target_u = Ueq.unsqueeze(1).to(self.device)
        target_v = Veq.unsqueeze(1).to(self.device)
        target_w = Weq.unsqueeze(1).to(self.device)

        pred_u, pred_v, pred_w = output[:, 0:1], output[:, 1:2], output[:, 2:3]

        loss_u = self.criterion(pred_u, target_u, mask_b)
        loss_v = self.criterion(pred_v, target_v, mask_b)
        loss_w = self.criterion(pred_w, target_w, mask_b)
        loss = loss_u + loss_v + loss_w
        return loss, img1.size(0)

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        loss, valid_count = self._process_train_batch(batch, batch_idx)
        if loss is not None:
            self.log("train_loss", loss, prog_bar=False, sync_dist=True)
            self.log("train_valid_samples", valid_count,
                     prog_bar=False, sync_dist=True)
            return loss
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def validation_step(self, batch, batch_idx):
        loss, valid_count = self._process_val_batch(batch, batch_idx)
        if loss is not None:
            self.log("val_loss", loss, prog_bar=False, sync_dist=True)
            self.log("val_valid_samples", valid_count,
                     prog_bar=False, sync_dist=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.Tmax, eta_min=1e-8
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    def on_train_epoch_start(self):
        self.epoch_nan_inf_count = 0

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get(
            "train_loss", torch.tensor(0.0)
        )
        self.train_losses.append(train_loss.item())
        if self.global_rank == 0:
            current_epoch = self.trainer.current_epoch
            print(f"Epoch {current_epoch}: {self.epoch_nan_inf_count} "
                  f"samples with NaN/Inf", flush=True)
        self.log("nan_inf_count", self.epoch_nan_inf_count,
                 prog_bar=False, sync_dist=True)

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get(
            "val_loss", torch.tensor(0.0)
        )
        self.val_losses.append(val_loss.item())


# ======================================================================
# Progress callback (mirrors cavity SimplifiedOutputCallback)
# ======================================================================

class SimplifiedOutputCallback(Callback):
    def __init__(self, num_epochs):
        super().__init__()
        self.num_epochs = num_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics.get(
            "train_loss", torch.tensor(0.0)
        ).item()
        val_loss = trainer.callback_metrics.get(
            "val_loss", torch.tensor(0.0)
        ).item()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] Epoch [{epoch + 1}/{self.num_epochs}] "
              f"Training Loss: {train_loss:.10f}, "
              f"Validation Loss: {val_loss:.10f}", flush=True)