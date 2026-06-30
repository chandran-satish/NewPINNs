"""
PyClaw-backed solver for the 1-D variable-velocity advection equation
======================================================================

    q_t + u(x; α) q_x = 0,     u(x; α) = α + sin(2πx)

The parameter α ∈ [1.5, 2.5] sets the mean advection speed; the
sin(2πx) term creates spatial stretching and compression.

Grid convention
---------------
Cell-centred finite-volume grid on [x0, x1] with nx cells:

    x_i = x0 + (i + 0.5) * dx,   i = 0, …, nx-1

Public API  (mirrors AllenCahnSolver used by the rest of NewPINNs)
------------------------------------------------------------------
solve(u_tn, alpha)                  – batched advance by n_i × dt
solve_sequential(ic, alpha, n_steps) – chain n_steps advances, return all states
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from clawpack import pyclaw, riemann


class VCAdvection1DSolver:
    """Advance q_t + (α + sin(2πx)) q_x = 0 by n_i × dt using PyClaw.

    Parameters
    ----------
    x0, x1     : float  domain extents
    n_points   : int    number of spatial cells  (matches ``spatial.n_points`` in config)
    dt         : float  macro time-step
    n_i        : int    solver steps per network interval
    """

    def __init__(
        self,
        x0: float,
        x1: float,
        n_points: int,
        dt: float,
        n_i: int,
        # Extra kwargs accepted but ignored so callers can pass sweeps_per_step etc.
        **_kwargs,
    ) -> None:
        self.x0, self.x1 = x0, x1
        self.nx = n_points
        self.dt = dt
        self.n_i = n_i
        self.total_time = n_i * dt

    # ── helpers ────────────────────────────────────────────────────

    @property
    def cell_centers(self) -> torch.Tensor:
        """Cell-centre coordinate vector, shape (nx,)."""
        dx = (self.x1 - self.x0) / self.nx
        return torch.linspace(
            self.x0 + 0.5 * dx,
            self.x1 - 0.5 * dx,
            self.nx,
            dtype=torch.float64,
        )

    @staticmethod
    def _velocity(xc: np.ndarray, alpha: float) -> np.ndarray:
        """u(x; α) = α + sin(2πx)."""
        return alpha + np.sin(2.0 * np.pi * xc)

    # ── single-sample solve ────────────────────────────────────────

    def _solve_single(self, q_np: np.ndarray, alpha: float) -> np.ndarray:
        """Advance one ``(N,)`` state by ``total_time``.

        Parameters
        ----------
        q_np  : ndarray, shape ``(N,)``, dtype float64
        alpha : float

        Returns
        -------
        ndarray, shape ``(N,)``
        """
        # ── Riemann solver and BCs (periodic) ────────────────────
        solver = pyclaw.ClawSolver1D(riemann.vc_advection_1D_py.vc_advection_1D)
        solver.kernel_language = "Python"
        solver.limiters        = pyclaw.limiters.tvd.MC
        solver.bc_lower[0]     = pyclaw.BC.periodic
        solver.bc_upper[0]     = pyclaw.BC.periodic
        solver.aux_bc_lower[0] = pyclaw.BC.periodic
        solver.aux_bc_upper[0] = pyclaw.BC.periodic

        # ── Domain and state ─────────────────────────────────────
        x_dim  = pyclaw.Dimension(self.x0, self.x1, self.nx, name="x")
        domain = pyclaw.Domain(x_dim)
        state  = pyclaw.State(domain, num_eqn=1, num_aux=1)

        state.q[0, :]   = q_np
        state.aux[0, :] = self._velocity(state.grid.x.centers, alpha)

        # ── Controller ───────────────────────────────────────────
        claw = pyclaw.Controller()
        claw.verbosity        = 0
        claw.output_format    = None
        claw.keep_copy        = True
        claw.num_output_times = 1
        claw.tfinal           = float(self.total_time)
        claw.solution         = pyclaw.Solution(state, domain)
        claw.solver           = solver

        try:
            claw.run()
            return claw.frames[-1].q[0].copy()   # shape (N,)
        except Exception as exc:                 # noqa: BLE001
            warnings.warn(
                f"PyClaw advection solve failed (α={alpha:.4f}): {exc}. "
                "Returning input unchanged as fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            return q_np.copy()

    # ── batched public API ─────────────────────────────────────────

    @torch.no_grad()
    def solve(
        self,
        u_tn: torch.Tensor,   # (B, N)
        alpha: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """Advance each sample by ``n_i × dt``.

        Parameters
        ----------
        u_tn  : Tensor ``(B, N)``
        alpha : Tensor ``(B,)``

        Returns
        -------
        Tensor ``(B, N)``
        """
        device = u_tn.device
        dtype  = u_tn.dtype
        q_np   = u_tn.detach().cpu().double().numpy()
        a_np   = alpha.detach().cpu().double().numpy().ravel()

        results = np.empty_like(q_np)
        for i in range(q_np.shape[0]):
            results[i] = self._solve_single(q_np[i], float(a_np[i]))

        return torch.tensor(results, device=device, dtype=dtype)

    @torch.no_grad()
    def solve_sequential(
        self,
        ic: torch.Tensor,  # (N,)
        alpha: float,
        n_steps: int,
    ) -> list[torch.Tensor]:
        """Chain *n_steps* solver calls and return all states including IC.

        Parameters
        ----------
        ic      : Tensor ``(N,)``
        alpha   : float
        n_steps : int

        Returns
        -------
        list[Tensor ``(N,)``] of length ``n_steps + 1``
        """
        q_np = ic.detach().cpu().double().numpy()
        states = [torch.tensor(q_np, dtype=torch.float64)]

        for _ in range(n_steps):
            q_np = self._solve_single(q_np, alpha)
            states.append(torch.tensor(q_np.copy(), dtype=torch.float64))

        return states