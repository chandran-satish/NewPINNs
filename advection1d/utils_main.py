#!/usr/bin/env python3
"""
NewPINNs – 1-D variable-velocity advection training entry point.

PDE:  q_t + (α + sin(2πx)) q_x = 0,   α ∈ [1.5, 2.5]

Usage
-----
    python main.py --config config.yaml --save
    python main.py --config config.yaml
    python main.py --config config.yaml --ckpt_path checkpoints/last.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import yaml
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

from NeurIPS_August.ad1d.utils_solver import VCAdvection1DSolver
from NeurIPS_August.ad1d.utils_model  import Net, NewPINNsModule
from NeurIPS_August.ad1d.utils_dataset import NewPINNsDataModule

torch.set_float32_matmul_precision("medium")


class EpochLogCallback(Callback):
    """Print train (and optionally val) loss at every epoch end."""

    def __init__(self, num_epochs: int) -> None:
        super().__init__()
        self.num_epochs = num_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return

        epoch   = trainer.current_epoch
        metrics = trainer.callback_metrics
        train_loss   = metrics.get("train/loss",        torch.tensor(0.0)).item()
        solver_loss  = metrics.get("train/solver_loss", torch.tensor(0.0)).item()
        ic_loss      = metrics.get("train/ic_loss",     torch.tensor(0.0)).item()
        val_loss     = metrics.get("val/loss", None)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        msg = (f"[{ts}] Epoch [{epoch + 1}/{self.num_epochs}] "
               f"Loss: {train_loss:.6e}  "
               f"(solver: {solver_loss:.6e}, IC: {ic_loss:.6e})")
        if val_loss is not None:
            msg += f", Val: {val_loss.item():.6e}"
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        msg += f", LR: {lr:.2e}"
        print(msg)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_solver(cfg: dict) -> VCAdvection1DSolver:
    tc = cfg["time"]
    sc = cfg["spatial"]
    return VCAdvection1DSolver(
        x0=sc["x0"],
        x1=sc["x1"],
        n_points=sc["n_points"],
        dt=tc["dt"],
        n_i=tc["n_i"],
    )


def build_net(cfg: dict) -> Net:
    nc = cfg["network"]
    return Net(
        input_dim=nc["input_dim"],
        output_dim=nc["output_dim"],
        hidden_dim=nc["hidden_dim"],
        num_layers=nc["num_layers"],
        activation=nc["activation"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="NewPINNs – 1D advection")
    parser.add_argument("--config",    type=str, default="config.yaml")
    parser.add_argument("--save",      action="store_true",
                        help="Save bare network weights after training")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Lightning checkpoint to resume from")
    args = parser.parse_args()

    cfg       = load_config(args.config)
    train_cfg = cfg["training"]

    # ── data ───────────────────────────────────────────────────────
    dm = NewPINNsDataModule(cfg)
    dm.setup()
    train_data = dm.train_data

    # ── solver + network + lightning module ─────────────────────────
    solver = build_solver(cfg)
    net    = build_net(cfg)
    module = NewPINNsModule(
        net=net,
        solver=solver,
        first_time=train_data["first_time"],
        ic_target=train_data["ic_target"],
        cfg=cfg,
    )

    # ── callbacks ──────────────────────────────────────────────────
    callbacks = [EpochLogCallback(num_epochs=train_cfg["max_epochs"])]

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Always keep the latest checkpoint
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

    # Periodic snapshots (optional)
    save_every = train_cfg.get("save_every_n_epochs", 0)
    if save_every > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                every_n_epochs=save_every,
                save_top_k=-1,
                monitor=None,
                filename="newpinns-advection-{epoch:05d}",
            )
        )

    # ── trainer ────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        log_every_n_steps=train_cfg.get("log_every_n_steps", 1),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        callbacks=callbacks,
        accelerator="auto",
        devices=1,
        strategy="ddp",
        enable_progress_bar=False,
    )

    trainer.fit(module, datamodule=dm, ckpt_path=args.ckpt_path)

    # ── save bare weights ──────────────────────────────────────────
    if args.save:
        model_path = Path(train_cfg.get("model_path", "newpinns_advection_weights.pth"))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), model_path)
        print(f"Network weights saved to {model_path}")


if __name__ == "__main__":
    main()