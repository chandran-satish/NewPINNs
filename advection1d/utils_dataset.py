"""
HDF5-backed datasets and Lightning DataModule for NewPINNs.

Expected h5 layout (written by ``generate_data.py``):
    alphas       (R,)
    time_points  (T,)
    xset         (N,)
    solutions    (R, T, N)
"""

from __future__ import annotations

from typing import Any

import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl


# ═══════════════════════════════════════════════════════════════════
#  Helpers for building grids from h5 metadata
# ═══════════════════════════════════════════════════════════════════

def _load_h5(path: str) -> dict[str, torch.Tensor]:
    """Read an h5 dataset into a dict of float32 tensors."""
    with h5py.File(path, "r") as f:
        return {
            "alphas": torch.tensor(f["alphas"][:], dtype=torch.float32),
            "time_points": torch.tensor(f["time_points"][:], dtype=torch.float32),
            "xset": torch.tensor(f["xset"][:], dtype=torch.float32),
            "solutions": torch.tensor(f["solutions"][:], dtype=torch.float32),
        }


def build_train_data(h5_path: str) -> dict[str, Any]:
    """Build NewPINNs training tensors from a training h5 file.

    Returns
    -------
    dict with:
        pairs       – (P, 2, N, 3)   consecutive-time input pairs
        first_time  – (R·N, 3)       network inputs at t = 0
        ic_target   – (R·N, 1)       initial-condition values from the h5
        time_points – (T,)
        xset        – (N,)
        alphas      – (R,)
    """
    d = _load_h5(h5_path)
    alphas = d["alphas"]          # (R,)
    time_pts = d["time_points"]   # (T,)
    xset = d["xset"]              # (N,)
    sols = d["solutions"]         # (R, T, N)

    R = len(alphas)
    T = len(time_pts)

    # ── full (T, R, N, 3) input tensor ─────────────────────────────
    t_g, a_g, x_g = torch.meshgrid(time_pts, alphas, xset, indexing="ij")
    inputs = torch.stack([t_g, a_g, x_g], dim=-1)          # (T, R, N, 3)

    # Flatten to (T·R, N, 3)
    total = inputs.view(T * R, -1, 3)

    # ── consecutive-time pairs ─────────────────────────────────────
    a = total[: -R]
    b = total[R:]
    pairs = torch.stack([a, b], dim=1)                      # (P, 2, N, 3)

    # ── initial-condition data (from h5 solutions at t=0) ──────────
    first_time_flat = total[:R].reshape(-1, 3)              # (R·N, 3)
    ic_target = sols[:, 0, :].reshape(-1, 1)                # (R·N, 1)

    return dict(
        pairs=pairs,
        first_time=first_time_flat,
        ic_target=ic_target,
        time_points=time_pts,
        xset=xset,
        alphas=alphas,
    )


def build_val_data(h5_path: str) -> dict[str, torch.Tensor]:
    """Build pointwise (input, target) tensors from a validation h5 file.

    Returns
    -------
    dict with:
        val_inputs  – (V, 3)  network inputs  (t, α, x)
        val_targets – (V, 1)  ground-truth u(x, t; α)
    """
    d = _load_h5(h5_path)
    alphas = d["alphas"]
    time_pts = d["time_points"]
    xset = d["xset"]
    sols = d["solutions"]         # (R, T, N)

    # Build a full meshgrid of inputs
    t_g, a_g, x_g = torch.meshgrid(time_pts, alphas, xset, indexing="ij")
    inputs = torch.stack([t_g, a_g, x_g], dim=-1)          # (T, R, N, 3)

    # Targets: sols is (R, T, N) → transpose to (T, R, N) to match grid
    targets = sols.permute(1, 0, 2)                         # (T, R, N)

    return dict(
        val_inputs=inputs.reshape(-1, 3),
        val_targets=targets.reshape(-1, 1),
    )


# ═══════════════════════════════════════════════════════════════════
#  PyTorch Datasets
# ═══════════════════════════════════════════════════════════════════

class PairDataset(Dataset):
    """Thin wrapper around the pre-computed pairs tensor."""

    def __init__(self, pairs: torch.Tensor) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return self.pairs.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.pairs[idx]


class ValidationDataset(Dataset):
    """Pointwise (input, target) dataset for data-driven validation."""

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


# ═══════════════════════════════════════════════════════════════════
#  Lightning DataModule
# ═══════════════════════════════════════════════════════════════════

class NewPINNsDataModule(pl.LightningDataModule):
    """Loads train / val data from h5 files specified in the config."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_data: dict[str, Any] | None = None
        self.val_data: dict[str, torch.Tensor] | None = None

    def setup(self, stage: str | None = None) -> None:
        dc = self.cfg["data"]

        if self.train_data is None:
            print(f"Loading training data from {dc['train_file']} …")
            self.train_data = build_train_data(dc["train_file"])
            R = len(self.train_data["alphas"])
            T = len(self.train_data["time_points"])
            P = self.train_data["pairs"].shape[0]
            print(f"  → {R} α values, {T} time-points, {P} training pairs")

        if self.val_data is None:
            print(f"Loading validation data from {dc['val_file']} …")
            self.val_data = build_val_data(dc["val_file"])
            V = self.val_data["val_inputs"].shape[0]
            print(f"  → {V} validation points")

    def train_dataloader(self) -> DataLoader:
        assert self.train_data is not None
        ds = PairDataset(self.train_data["pairs"])
        return DataLoader(
            ds,
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_data is not None
        ds = ValidationDataset(
            self.val_data["val_inputs"],
            self.val_data["val_targets"],
        )
        return DataLoader(
            ds,
            batch_size=self.cfg["training"]["batch_size"] * 4,
            shuffle=False,
            num_workers=0,
        )