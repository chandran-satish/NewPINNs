"""
PyClaw-backed solver for the 2-D shallow water equations
=========================================================

    h_t  + (hu)_x + (hv)_y                     = 0
    (hu)_t + (hu² + ½gh²)_x + (huv)_y          = 0
    (hv)_t + (huv)_x + (hv² + ½gh²)_y          = 0

IC:  Radial dam break — circular region of depth h_in, surroundings h_out.
Parameter: h_in (inner dam height) ∈ [1.5, 3.0].

Public API
----------
solve(q_tn, h_ins)                     – batched advance by n_i × dt
solve_sequential(ic, n_steps)          – chain n_steps, return all states
"""

from __future__ import annotations

import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from clawpack import pyclaw, riemann
from clawpack.riemann.shallow_roe_with_efix_2D_constants import (
    depth, x_momentum, y_momentum, num_eqn,
)

logging.getLogger("pyclaw").setLevel(logging.WARNING)


# ── module-level solve function (picklable) ────────────────────────

def _pyclaw_swe2d_single(args: tuple) -> np.ndarray:
    """Advance one (3, ny, nx) state by total_time.

    Parameters
    ----------
    args : (q_np, grav, x0, x1, y0, y1, nx, ny, total_time, riemann_name)

    Returns
    -------
    ndarray (3, ny, nx) or NaN array on failure
    """
    q_np, grav, x0, x1, y0, y1, nx, ny, total_time, riemann_name = args

    logging.getLogger("pyclaw").setLevel(logging.WARNING)

    _NAN_RESULT = np.full_like(q_np, np.nan)

    # ── pre-flight checks ─────────────────────────────────────────
    if np.isnan(q_np).any() or np.isinf(q_np).any():
        return _NAN_RESULT
    if (q_np[0] <= 0).any():        # depth must be positive
        return _NAN_RESULT

    # ── Riemann solver ────────────────────────────────────────────
    if riemann_name == "roe":
        rs = riemann.shallow_roe_with_efix_2D
    else:
        rs = riemann.shallow_hlle_2D

    solver = pyclaw.ClawSolver2D(rs)
    solver.limiters = pyclaw.limiters.tvd.MC
    solver.dimensional_split = 1

    solver.bc_lower[0] = pyclaw.BC.extrap
    solver.bc_upper[0] = pyclaw.BC.wall
    solver.bc_lower[1] = pyclaw.BC.extrap
    solver.bc_upper[1] = pyclaw.BC.wall

    # ── domain + state ────────────────────────────────────────────
    x_dim  = pyclaw.Dimension(x0, x1, nx, name="x")
    y_dim  = pyclaw.Dimension(y0, y1, ny, name="y")
    domain = pyclaw.Domain([x_dim, y_dim])
    state  = pyclaw.State(domain, num_eqn)

    state.problem_data["grav"] = float(grav)

    # Our array: (3, ny, nx) → PyClaw: q[eqn, ix, jy]
    for eq in range(num_eqn):
        state.q[eq, :, :] = q_np[eq].T

    # ── controller ────────────────────────────────────────────────
    claw = pyclaw.Controller()
    claw.verbosity        = 0
    claw.output_format    = None
    claw.keep_copy        = True
    claw.num_output_times = 1
    claw.tfinal           = float(total_time)
    claw.solution         = pyclaw.Solution(state, domain)
    claw.solver           = solver

    try:
        claw.run()
        q_out = claw.frames[-1].q.copy()

        if np.isnan(q_out).any() or np.isinf(q_out).any():
            return _NAN_RESULT

        result = np.empty_like(q_np)
        for eq in range(num_eqn):
            result[eq] = q_out[eq].T
        return result

    except Exception as exc:
        warnings.warn(
            f"PyClaw SWE-2D solve failed: {exc}.",
            RuntimeWarning, stacklevel=2,
        )
        return _NAN_RESULT


# ═══════════════════════════════════════════════════════════════════

class SWE2DSolver:
    """Advance 2-D shallow water equations by n_i × dt using PyClaw.

    Parameters
    ----------
    x0, x1, y0, y1 : float    domain extents
    nx, ny          : int      cells in x, y
    dt              : float    macro time-step
    n_i             : int      sub-steps per macro step
    grav            : float    gravitational acceleration
    max_workers     : int      parallel workers
    use_threads     : bool     thread vs process pool
    riemann_solver  : str      'roe' or 'hlle'
    """

    def __init__(
        self,
        x0: float, x1: float,
        y0: float, y1: float,
        nx: int, ny: int,
        dt: float,
        n_i: int,
        grav: float = 1.0,
        max_workers: int | None = None,
        use_threads: bool = True,
        riemann_solver: str = "roe",
        **_kwargs,
    ) -> None:
        self.x0, self.x1 = x0, x1
        self.y0, self.y1 = y0, y1
        self.nx, self.ny  = nx, ny
        self.dt           = dt
        self.n_i          = n_i
        self.total_time   = n_i * dt
        self.grav         = grav
        self.riemann_name = riemann_solver.lower()

        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, 16)
        self.max_workers = max_workers

        self._pool = None
        if max_workers > 1:
            PoolCls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
            self._pool = PoolCls(max_workers=max_workers)

    # ── helpers ────────────────────────────────────────────────────

    @property
    def cell_centers_x(self) -> torch.Tensor:
        dx = (self.x1 - self.x0) / self.nx
        return torch.linspace(self.x0 + 0.5 * dx, self.x1 - 0.5 * dx, self.nx,
                              dtype=torch.float64)

    @property
    def cell_centers_y(self) -> torch.Tensor:
        dy = (self.y1 - self.y0) / self.ny
        return torch.linspace(self.y0 + 0.5 * dy, self.y1 - 0.5 * dy, self.ny,
                              dtype=torch.float64)

    def _make_args(self, q_np: np.ndarray) -> tuple:
        return (q_np, self.grav,
                self.x0, self.x1, self.y0, self.y1,
                self.nx, self.ny, self.total_time, self.riemann_name)

    # ── batched public API ─────────────────────────────────────────

    @torch.no_grad()
    def solve(
        self,
        q_tn: torch.Tensor,      # (B, 3, ny, nx)
        h_ins: torch.Tensor,     # (B,) — not used by solver, API compat
    ) -> torch.Tensor:
        """Advance each sample by n_i × dt.

        h_in only affects the IC, not the PDE step.
        """
        device = q_tn.device
        dtype  = q_tn.dtype
        q_np   = q_tn.detach().cpu().double().numpy()
        B      = q_np.shape[0]

        args = [self._make_args(q_np[i]) for i in range(B)]

        if B == 1 or self._pool is None:
            results_list = [_pyclaw_swe2d_single(a) for a in args]
        else:
            results_list = list(self._pool.map(_pyclaw_swe2d_single, args))

        results = np.stack(results_list, axis=0)
        return torch.tensor(results, device=device, dtype=dtype)

    @torch.no_grad()
    def solve_sequential(
        self,
        ic: torch.Tensor,        # (3, ny, nx)
        n_steps: int,
    ) -> list[torch.Tensor]:
        """Chain n_steps advances; return all states including IC."""
        q_np   = ic.detach().cpu().double().numpy()
        states = [torch.tensor(q_np.copy(), dtype=torch.float64)]

        for _ in range(n_steps):
            args = self._make_args(q_np)
            q_np = _pyclaw_swe2d_single(args)
            states.append(torch.tensor(q_np.copy(), dtype=torch.float64))

        return states

    # ── sanity check ───────────────────────────────────────────────

    def plot_trajectory(
        self,
        h_in: float,
        n_steps: int,
        cfg: dict,
        out_path: str = "swe2d_sanity.png",
    ) -> None:
        ic = dam_break_ic(
            self.cell_centers_x, self.cell_centers_y,
            h_in=h_in, cfg=cfg,
        )
        states = self.solve_sequential(ic, n_steps)

        ncols = min(len(states), 6)
        nrows = (len(states) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        axes = np.atleast_1d(axes).ravel()

        for i, s in enumerate(states):
            if i >= len(axes):
                break
            im = axes[i].pcolormesh(s[0].numpy(), cmap="RdBu_r")
            axes[i].set_title(f"t = {i * self.total_time:.2f}")
            axes[i].set_aspect("equal")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"SWE dam break  h_in = {h_in:.2f}", fontsize=12)
        plt.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close("all")
        print(f"Saved {out_path}")


# ═══════════════════════════════════════════════════════════════════
#  Initial condition — radial dam break
# ═══════════════════════════════════════════════════════════════════

def dam_break_ic(
    xc: torch.Tensor,       # (nx,)
    yc: torch.Tensor,       # (ny,)
    h_in: float,
    cfg: dict,
) -> torch.Tensor:
    """Build (3, ny, nx) conserved-variable IC for the radial dam break.

    Parameters
    ----------
    xc, yc : cell centers
    h_in   : water depth inside the dam
    cfg    : config dict (needs physics.h_out, physics.dam_radius)

    Returns
    -------
    (3, ny, nx): (h, hu, hv)
    """
    phys = cfg["physics"]
    h_out      = phys["h_out"]
    dam_radius = phys["dam_radius"]

    yy, xx = torch.meshgrid(yc, xc, indexing="ij")
    xx = xx.double()
    yy = yy.double()

    r = torch.sqrt(xx ** 2 + yy ** 2)

    h  = torch.where(r <= dam_radius, h_in, h_out).double()
    hu = torch.zeros_like(h)
    hv = torch.zeros_like(h)

    return torch.stack([h, hu, hv], dim=0)    # (3, ny, nx)


# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    sc = cfg["spatial"]
    tc = cfg["time"]
    n_steps = round((tc["t1"] - tc["t0"]) / tc["dt"])

    solver = SWE2DSolver(
        x0=sc["x0"], x1=sc["x1"], y0=sc["y0"], y1=sc["y1"],
        nx=sc["nx"], ny=sc["ny"],
        dt=tc["dt"], n_i=tc["n_i"],
        grav=cfg["physics"]["grav"],
        max_workers=1,
    )

    for h in [1.5, 2.0, 3.0]:
        solver.plot_trajectory(h, n_steps, cfg,
                               out_path=f"swe2d_sanity_h{h:.1f}.png")