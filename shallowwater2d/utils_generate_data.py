#!/usr/bin/env python3
"""
Generate HDF5 datasets for NewPINNs – 2-D shallow water radial dam break.

Three datasets with non-overlapping h_in values:
  1. Pretrain  (16)  — Phase 1 supervised warm-start
  2. Train    (256)  — Phase 2 solver-consistency
  3. Val       (32)  — held-out evaluation

Edit the parameters below, then run:  python generate_data.py
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import torch
import yaml

from NeurIPS_August.swe2d.utils_solver import SWE2DSolver, dam_break_ic

CONFIG_FILE = "config.yaml"

H_IN_MIN = 1.5
H_IN_MAX = 3.0

N_PRETRAIN = 16
N_TRAIN    = 256
N_VAL      = 32

PRETRAIN_SEED = 7
TRAIN_SEED    = 42
VAL_SEED      = 123

PRETRAIN_FILE = "./data/swe2d_pretrain.h5"
TRAIN_FILE    = "./data/swe2d_train.h5"
VAL_FILE      = "./data/swe2d_val.h5"


def generate_h5(
    out_path: str,
    h_ins: np.ndarray,
    solver: SWE2DSolver,
    time_points: torch.Tensor,
    xc: torch.Tensor,
    yc: torch.Tensor,
    cfg: dict,
) -> None:
    T   = len(time_points)
    R   = len(h_ins)
    ny  = len(yc)
    nx  = len(xc)

    solutions = np.empty((R, T, 3, ny, nx), dtype=np.float64)

    for i, h in enumerate(h_ins):
        ic     = dam_break_ic(xc, yc, h_in=float(h), cfg=cfg)
        states = solver.solve_sequential(ic, n_steps=T - 1)
        for t_idx, state in enumerate(states):
            solutions[i, t_idx] = state.numpy()
        print(f"  h_in = {h:.4f}  ({i + 1}/{R})")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("h_ins",       data=h_ins.astype(np.float64))
        f.create_dataset("time_points", data=time_points.numpy().astype(np.float64))
        f.create_dataset("xc",          data=xc.numpy().astype(np.float64))
        f.create_dataset("yc",          data=yc.numpy().astype(np.float64))
        f.create_dataset("solutions",   data=solutions)

    print(f"\n  → Wrote {out_path}  (R={R}, T={T}, 3×{ny}×{nx})\n")


def main() -> None:
    with open(CONFIG_FILE) as f:
        cfg = yaml.safe_load(f)

    sc = cfg["spatial"]
    tc = cfg["time"]

    n_steps     = round((tc["t1"] - tc["t0"]) / tc["dt"])
    time_points = torch.linspace(tc["t0"], tc["t1"], steps=n_steps + 1)

    solver = SWE2DSolver(
        x0=sc["x0"], x1=sc["x1"], y0=sc["y0"], y1=sc["y1"],
        nx=sc["nx"], ny=sc["ny"],
        dt=tc["dt"], n_i=tc["n_i"],
        grav=cfg["physics"]["grav"],
        max_workers=1,
    )

    xc = solver.cell_centers_x
    yc = solver.cell_centers_y

    pretrain_rng = np.random.default_rng(PRETRAIN_SEED)
    pretrain_h = np.sort(pretrain_rng.uniform(H_IN_MIN, H_IN_MAX, size=N_PRETRAIN))

    train_rng = np.random.default_rng(TRAIN_SEED)
    train_h = np.sort(train_rng.uniform(H_IN_MIN, H_IN_MAX, size=N_TRAIN))

    val_rng = np.random.default_rng(VAL_SEED)
    val_h = np.sort(val_rng.uniform(H_IN_MIN, H_IN_MAX, size=N_VAL))

    all_h = np.concatenate([pretrain_h, train_h, val_h])
    assert len(np.unique(all_h)) == len(all_h), "Duplicate h_in values — change seeds."

    print(f"Pretrain: {N_PRETRAIN} h_in values ∈ [{H_IN_MIN}, {H_IN_MAX}]")
    generate_h5(PRETRAIN_FILE, pretrain_h, solver, time_points, xc, yc, cfg)

    print(f"Training: {N_TRAIN} h_in values ∈ [{H_IN_MIN}, {H_IN_MAX}]")
    generate_h5(TRAIN_FILE, train_h, solver, time_points, xc, yc, cfg)

    print(f"Validation: {N_VAL} h_in values ∈ [{H_IN_MIN}, {H_IN_MAX}]")
    generate_h5(VAL_FILE, val_h, solver, time_points, xc, yc, cfg)

    print("Done.")


if __name__ == "__main__":
    main()