"""
HDF5-backed datasets and Lightning DataModule for NewPINNs – 2D SWE
radial dam break (U-Net).

H5 layout:  h_ins (R,), time_points (T,), xc (nx,), yc (ny,),
            solutions (R, T, 3, ny, nx)

U-Net:  input (B, 2, ny, nx) [t, h_in]  →  output (B, 3, ny, nx) [h, hu, hv]
"""

from __future__ import annotations
from typing import Any

import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl


def _load_h5(path: str) -> dict[str, torch.Tensor]:
    with h5py.File(path, "r") as f:
        return {
            "h_ins":       torch.tensor(f["h_ins"][:],       dtype=torch.float32),
            "time_points": torch.tensor(f["time_points"][:], dtype=torch.float32),
            "xc":          torch.tensor(f["xc"][:],          dtype=torch.float32),
            "yc":          torch.tensor(f["yc"][:],          dtype=torch.float32),
            "solutions":   torch.tensor(f["solutions"][:],   dtype=torch.float32),
        }


def _make_const_input(t_val: float, h_in_val: float, ny: int, nx: int) -> torch.Tensor:
    img = torch.empty(2, ny, nx, dtype=torch.float32)
    img[0].fill_(t_val)
    img[1].fill_(h_in_val)
    return img


def _build_pair_data(h5_path: str) -> dict[str, torch.Tensor]:
    d     = _load_h5(h5_path)
    h_ins = d["h_ins"]
    tpts  = d["time_points"]
    sols  = d["solutions"]

    R, T, _, ny, nx = sols.shape

    in_tn_list   = []
    in_tn1_list  = []
    param_list   = []
    tgt_tn1_list = []

    for ti in range(T - 1):
        for ri in range(R):
            in_tn_list.append(_make_const_input(tpts[ti].item(), h_ins[ri].item(), ny, nx))
            in_tn1_list.append(_make_const_input(tpts[ti + 1].item(), h_ins[ri].item(), ny, nx))
            param_list.append(h_ins[ri])
            tgt_tn1_list.append(sols[ri, ti + 1])

    ic_inputs = torch.stack([
        _make_const_input(tpts[0].item(), h_ins[ri].item(), ny, nx)
        for ri in range(R)
    ])

    return dict(
        inputs_tn=torch.stack(in_tn_list),
        inputs_tn1=torch.stack(in_tn1_list),
        params_flat=torch.stack(param_list),
        targets_tn1=torch.stack(tgt_tn1_list),
        ic_inputs=ic_inputs,
        ic_target=sols[:, 0],
        time_points=tpts,
        h_ins=h_ins,
    )


def build_val_data(h5_path: str) -> dict[str, torch.Tensor]:
    d     = _load_h5(h5_path)
    h_ins = d["h_ins"]
    tpts  = d["time_points"]
    sols  = d["solutions"]

    R, T, _, ny, nx = sols.shape

    inputs_list  = []
    targets_list = []
    for ti in range(T):
        for ri in range(R):
            inputs_list.append(
                _make_const_input(tpts[ti].item(), h_ins[ri].item(), ny, nx)
            )
            targets_list.append(sols[ri, ti])

    return dict(
        val_inputs=torch.stack(inputs_list),
        val_targets=torch.stack(targets_list),
    )


# ═══════════════════════════════════════════════════════════════════

class PairDataset(Dataset):
    def __init__(self, inputs_tn, inputs_tn1, params, targets) -> None:
        self.inputs_tn  = inputs_tn
        self.inputs_tn1 = inputs_tn1
        self.params     = params
        self.targets    = targets

    def __len__(self) -> int:
        return self.inputs_tn.size(0)

    def __getitem__(self, idx):
        return (self.inputs_tn[idx], self.inputs_tn1[idx],
                self.params[idx], self.targets[idx])


class ValidationDataset(Dataset):
    def __init__(self, inputs, targets) -> None:
        self.inputs  = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ═══════════════════════════════════════════════════════════════════

class NewPINNsDataModule(pl.LightningDataModule):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg           = cfg
        self.pretrain_data = None
        self.train_data    = None
        self.val_data      = None
        self.phase         = 1

    def setup(self, stage: str | None = None) -> None:
        dc = self.cfg["data"]

        if self.pretrain_data is None:
            print(f"Loading pretrain data from {dc['pretrain_file']} …")
            self.pretrain_data = _build_pair_data(dc["pretrain_file"])
            R = len(self.pretrain_data["h_ins"])
            P = self.pretrain_data["inputs_tn"].shape[0]
            print(f"  → {R} h_in values, {P} pretrain pairs")

        if self.train_data is None:
            print(f"Loading training data from {dc['train_file']} …")
            self.train_data = _build_pair_data(dc["train_file"])
            R = len(self.train_data["h_ins"])
            P = self.train_data["inputs_tn"].shape[0]
            print(f"  → {R} h_in values, {P} training pairs")

        if self.val_data is None:
            print(f"Loading validation data from {dc['val_file']} …")
            self.val_data = build_val_data(dc["val_file"])
            V = self.val_data["val_inputs"].shape[0]
            print(f"  → {V} validation points")

    def train_dataloader(self) -> DataLoader:
        td = self.pretrain_data if self.phase == 1 else self.train_data
        ds = PairDataset(td["inputs_tn"], td["inputs_tn1"],
                         td["params_flat"], td["targets_tn1"])
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