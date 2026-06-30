#!/usr/bin/env python3
"""
NewPINNs – 2-D shallow water radial dam break (U-Net).

PDE:    2D shallow water equations
IC:     Radial dam break
Param:  h_in (inner dam height) ∈ [1.5, 3.0]

Usage:  python main.py --config config.yaml --save
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import yaml
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

from NeurIPS_August.swe2d.utils_solver  import SWE2DSolver
from NeurIPS_August.swe2d.utils_model   import SWEUNet, NewPINNsModule
from NeurIPS_August.swe2d.utils_dataset import NewPINNsDataModule

torch.set_float32_matmul_precision("medium")


class EpochLogCallback(Callback):
    def __init__(self, num_epochs: int) -> None:
        super().__init__()
        self.num_epochs = num_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return

        epoch   = trainer.current_epoch
        metrics = trainer.callback_metrics
        train_loss  = metrics.get("train/loss",        torch.tensor(0.0)).item()
        solver_loss = metrics.get("train/solver_loss", torch.tensor(0.0)).item()
        ic_loss     = metrics.get("train/ic_loss",     torch.tensor(0.0)).item()
        phase       = metrics.get("train/phase",       torch.tensor(0.0)).item()
        val_loss    = metrics.get("val/loss", None)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        phase_str = "Phase2" if phase > 0.5 else "Phase1"
        msg = (f"[{ts}] Epoch [{epoch + 1}/{self.num_epochs}] [{phase_str}] "
               f"Loss: {train_loss:.6e}  "
               f"(solver: {solver_loss:.6e}, IC: {ic_loss:.6e})")
        if val_loss is not None:
            msg += f", Val: {val_loss.item():.6e}"
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        msg += f", LR: {lr:.2e}"
        nan_fb = metrics.get("train/nan_fallbacks", None)
        if nan_fb is not None:
            msg += f", NaN fallbacks: {nan_fb.item():.0f}"
        print(msg)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_solver(cfg: dict) -> SWE2DSolver:
    sc  = cfg["spatial"]
    tc  = cfg["time"]
    slv = cfg.get("solver", {})
    return SWE2DSolver(
        x0=sc["x0"], x1=sc["x1"],
        y0=sc["y0"], y1=sc["y1"],
        nx=sc["nx"], ny=sc["ny"],
        dt=tc["dt"],
        n_i=tc["n_i"],
        grav=cfg["physics"]["grav"],
        max_workers=slv.get("max_workers", None),
        use_threads=slv.get("use_threads", True),
        riemann_solver=slv.get("riemann_solver", "roe"),
    )


def build_net(cfg: dict) -> SWEUNet:
    return SWEUNet(cfg.get("unet", {}))


def main() -> None:
    parser = argparse.ArgumentParser(description="NewPINNs – 2D SWE Dam Break")
    parser.add_argument("--config",    type=str, default="config.yaml")
    parser.add_argument("--save",      action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=None)
    args = parser.parse_args()

    cfg       = load_config(args.config)
    train_cfg = cfg["training"]

    dm = NewPINNsDataModule(cfg)
    dm.setup()

    solver = build_solver(cfg)
    net    = build_net(cfg)

    pretrain = dm.pretrain_data
    module = NewPINNsModule(
        net=net,
        solver=solver,
        ic_inputs=pretrain["ic_inputs"],
        ic_target=pretrain["ic_target"],
        cfg=cfg,
    )

    callbacks = [EpochLogCallback(num_epochs=train_cfg["max_epochs"])]

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/swe2d"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks.append(
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="last",
            every_n_epochs=1,
            save_top_k=1,
            monitor=None,
            enable_version_counter=False,
        )
    )

    save_every = train_cfg.get("save_every_n_epochs", 0)
    if save_every > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                every_n_epochs=save_every,
                save_top_k=-1,
                monitor=None,
                filename="newpinns-swe2d-{epoch:05d}",
            )
        )

    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        log_every_n_steps=train_cfg.get("log_every_n_steps", 1),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        reload_dataloaders_every_n_epochs=1,
        callbacks=callbacks,
        accelerator="auto",
        devices=2,
        strategy="ddp",
        enable_progress_bar=False,
    )

    trainer.fit(module, datamodule=dm,
                ckpt_path=args.ckpt_path or train_cfg.get("ckpt_path", None))

    if args.save:
        model_path = Path(train_cfg.get("model_path", "newpinns_swe2d_weights.pth"))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), model_path)
        print(f"Network weights saved to {model_path}")


if __name__ == "__main__":
    main()