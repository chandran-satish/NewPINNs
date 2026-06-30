#!/usr/bin/env python3
"""
Generate HDF5 datasets for NewPINNs – 1-D variable-velocity advection.

PDE:   q_t + u(x; α) q_x = 0,   u(x; α) = α + sin(2πx)
IC:    q(x, 0) = exp(-50(x − 0.3)²)  +  1[0.6 < x < 0.8]
Domain: x ∈ [0, 1], t ∈ [0, 0.3],  α ∈ [1.5, 2.5]

Edit the parameters below, then run:  python generate_data.py
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import torch
import yaml

from NeurIPS_August.ad1d.utils_solver import VCAdvection1DSolver

# ═══════════════════════════════════════════════════════════════════
#  Parameters
# ═══════════════════════════════════════════════════════════════════

CONFIG_FILE = "config.yaml"

ALPHA_MIN = 1.5
ALPHA_MAX = 2.5

N_TRAIN    = 256
N_VAL      = 16

TRAIN_SEED = 42
VAL_SEED   = 123

TRAIN_FILE = f"./data/data_train_{N_TRAIN}.h5"
VAL_FILE   = f"./data/data_val_{N_VAL}.h5"

# ═══════════════════════════════════════════════════════════════════


def initial_condition(x: torch.Tensor) -> torch.Tensor:
    """q(x, 0) = Gaussian bump at 0.3 + top-hat on [0.6, 0.8].

    The combination of a smooth feature and a sharp discontinuity gives
    the network a challenging multi-scale IC to learn.
    """
    x_np = x.numpy()
    gaussian = np.exp(-50.0 * (x_np - 0.3) ** 2)
    tophat   = ((x_np > 0.6) & (x_np < 0.8)).astype(np.float64)
    return torch.tensor(gaussian + tophat, dtype=torch.float64)


def generate_h5(
    out_path: str,
    alphas: np.ndarray,
    solver: VCAdvection1DSolver,
    time_points: torch.Tensor,
    xset: torch.Tensor,
) -> None:
    """Roll out the solver for each α and write results to *out_path*."""
    T = len(time_points)
    N = len(xset)
    R = len(alphas)

    ic = initial_condition(xset)
    solutions = np.empty((R, T, N), dtype=np.float64)

    for i, alpha_val in enumerate(alphas):
        states = solver.solve_sequential(ic, float(alpha_val), n_steps=T - 1)
        for t_idx, state in enumerate(states):
            solutions[i, t_idx, :] = state.numpy()
        print(f"  α = {alpha_val:.4f}  ({i + 1}/{R})")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("alphas",      data=alphas.astype(np.float64))
        f.create_dataset("time_points", data=time_points.numpy().astype(np.float64))
        f.create_dataset("xset",        data=xset.numpy().astype(np.float64))
        f.create_dataset("solutions",   data=solutions)

    print(f"\n  → Wrote {out_path}  (R={R}, T={T}, N={N})\n")


def main() -> None:
    with open(CONFIG_FILE) as f:
        cfg = yaml.safe_load(f)

    tc = cfg["time"]
    sc = cfg["spatial"]

    # ── temporal grid ─────────────────────────────────────────────
    # t ∈ [0, 0.3] with dt = 0.05 → 7 time points
    n_steps = round((tc["t1"] - tc["t0"]) / tc["dt"])
    time_points = torch.linspace(tc["t0"], tc["t1"], steps=n_steps + 1)

    # Sub-sample by n_i if network skips solver steps
    if tc["n_i"] > 1:
        idx = torch.arange(len(time_points))
        time_points = time_points[idx % tc["n_i"] == 0]

    xset = torch.linspace(sc["x0"], sc["x1"], steps=sc["n_points"])

    # ── solver ────────────────────────────────────────────────────
    solver = VCAdvection1DSolver(
        x0=sc["x0"],
        x1=sc["x1"],
        n_points=sc["n_points"],
        dt=tc["dt"],
        n_i=tc["n_i"],
    )

    # ── sample alphas ─────────────────────────────────────────────
    train_rng    = np.random.default_rng(TRAIN_SEED)
    train_alphas = np.sort(train_rng.uniform(ALPHA_MIN, ALPHA_MAX, size=N_TRAIN))

    val_rng    = np.random.default_rng(VAL_SEED)
    val_alphas = np.sort(val_rng.uniform(ALPHA_MIN, ALPHA_MAX, size=N_VAL))

    # ── generate ──────────────────────────────────────────────────
    print(f"Training: {N_TRAIN} α values ∈ [{ALPHA_MIN}, {ALPHA_MAX}]  (seed={TRAIN_SEED})")
    generate_h5(TRAIN_FILE, train_alphas, solver, time_points, xset)

    print(f"Validation: {N_VAL} α values ∈ [{ALPHA_MIN}, {ALPHA_MAX}]  (seed={VAL_SEED})")
    generate_h5(VAL_FILE, val_alphas, solver, time_points, xset)

    print("Done.")


if __name__ == "__main__":
    main()