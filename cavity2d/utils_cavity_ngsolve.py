import numpy as np
import os
from ngsolve import *
from netgen.geom2d import unit_square
from ngsolve import x, y
from ngsolve.internal import *  # VoxelCoefficient lives here


class CavitySolver:
    """
    Persistent NGSolve solver for the 2D lid-driven cavity.
    
    Builds mesh, FE spaces, and bilinear-form structure ONCE.
    Call .solve() with different nu / initial conditions without
    re-creating the mesh or respawning a subprocess.
    
    Because the Stokes bilinear form depends on nu, the matrices
    are reassembled + refactorised whenever nu changes.
    """

    def __init__(self, maxh=0.05, order_v=3, order_p=2, tau=0.003, nx=32, ny=32, num_threads=None):
        # ---- threading ------------------------------------------------
        if num_threads is None:
            num_threads = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
        SetNumThreads(num_threads)

        # ---- mesh & spaces (created once) -----------------------------
        self.mesh = Mesh(unit_square.GenerateMesh(maxh=maxh)).Curve(3)
        V = VectorH1(self.mesh, order=order_v, dirichlet="top|bottom|left|right")
        Q = H1(self.mesh, order=order_p)
        self.X = V * Q

        self.u, self.p = self.X.TrialFunction()
        self.v, self.q = self.X.TestFunction()

        self.tau = tau
        self.nx = nx
        self.ny = ny

        # ---- grid points for sampling (created once) ------------------
        x_pts = np.linspace(0, 1, nx)
        y_pts = np.linspace(0, 1, ny)
        self._mesh_pts = [self.mesh(xi, yi) for yi in y_pts for xi in x_pts]

        # ---- convection form (parameter-independent, nonassemble) -----
        self.conv = BilinearForm(self.X, nonassemble=True)
        self.conv += (Grad(self.u) * self.u) * self.v * dx

        # ---- cached nu for avoiding redundant reassembly --------------
        self._cached_nu = None
        self._a = None
        self._inv = None

        # ---- reusable GridFunctions -----------------------------------
        self.gfu = GridFunction(self.X)
        self.gfu_int = GridFunction(self.X)

    # ------------------------------------------------------------------
    def _assemble(self, nu):
        """Assemble stiffness & time-stepper matrices for a given nu."""
        if nu == self._cached_nu:
            return
        u, v, p, q = self.u, self.v, self.p, self.q
        stokes = (nu * InnerProduct(grad(u), grad(v))
                  + div(u) * q + div(v) * p
                  - 1e-10 * p * q) * dx
        self._a = BilinearForm(stokes).Assemble()

        mstar = BilinearForm(u * v * dx + self.tau * stokes).Assemble()
        self._inv = mstar.mat.Inverse(self.X.FreeDofs(), inverse="sparsecholesky")
        self._cached_nu = nu

    # ------------------------------------------------------------------
    def solve(self, nu, uin_max, t_iter, U_initial, V_initial, P_initial):
        """
        Run the cavity solver.

        Parameters
        ----------
        nu : float
            Kinematic viscosity (1/Re).
        uin_max : float
            Lid velocity magnitude.
        t_iter : int
            Number of pseudo-time iterations.
        U_initial, V_initial, P_initial : ndarray (ny, nx)
            Initial fields on the uniform grid.

        Returns
        -------
        U, V, P : ndarray (ny, nx)
            Solution fields sampled on the uniform grid.
        """
        self._assemble(nu)

        # ---- set interior initial condition ---------------------------
        U64 = np.float64(U_initial)
        V64 = np.float64(V_initial)
        P64 = np.float64(P_initial)

        ufunc = VoxelCoefficient((0, 0), (1, 1), U64, linear=True)
        vfunc = VoxelCoefficient((0, 0), (1, 1), V64, linear=True)
        pfunc = VoxelCoefficient((0, 0), (1, 1), P64, linear=True)

        self.gfu_int.components[0].Set(CoefficientFunction((ufunc, vfunc)))
        self.gfu_int.components[1].Set(pfunc)

        # ---- set boundary condition (lid) on gfu ----------------------
        self.gfu.vec[:] = 0.0
        lid_velocity = CoefficientFunction((uin_max, 0))
        self.gfu.components[0].Set(lid_velocity,
                                   definedon=self.mesh.Boundaries("top"))

        # ---- copy interior DOFs from gfu_int --------------------------
        free_dofs = self.X.FreeDofs()
        for dof in range(self.X.ndof):
            if free_dofs[dof]:
                self.gfu.vec[dof] = self.gfu_int.vec[dof]

        # ---- pseudo-time iteration ------------------------------------
        with TaskManager():
            for _ in range(t_iter):
                res = self.conv.Apply(self.gfu.vec) + self._a.mat * self.gfu.vec
                self.gfu.vec.data -= self.tau * self._inv * res

        # ---- sample on uniform grid -----------------------------------
        U, V, P = self._sample()
        return U, V, P

    # ------------------------------------------------------------------
    def _sample(self):
        """Sample velocity & pressure on the pre-built uniform grid."""
        vel = self.gfu.components[0]
        pres = self.gfu.components[1]

        uv = np.array([vel(mp) for mp in self._mesh_pts])
        pp = np.array([pres(mp) for mp in self._mesh_pts])

        U = uv[:, 0].reshape(self.ny, self.nx)
        V = uv[:, 1].reshape(self.ny, self.nx)
        P = pp.reshape(self.ny, self.nx)
        return U, V, P


# ======================================================================
# Convenience wrappers (backward-compatible API for data generation, etc.)
# ======================================================================

def run_and_capture_ngsolve(nu=0.001, uin_max=1.0, tau=0.001, t_iter=1000,
                            U_initial=None, V_initial=None, P_initial=None, tag=""):
    """Drop-in replacement that uses a module-level persistent solver."""
    global _MODULE_SOLVER
    if '_MODULE_SOLVER' not in globals() or _MODULE_SOLVER is None:
        _MODULE_SOLVER = CavitySolver(tau=tau)
    U, V, P = _MODULE_SOLVER.solve(nu, uin_max, t_iter, U_initial, V_initial, P_initial)
    return U, V, P, None


def run_ngsolve_custom(nu=0.001, uin_max=1.0, tau=0.001, t_iter=1000,
                       U_initial=None, V_initial=None, P_initial=None, conn=None, tag=""):
    """Backward-compatible wrapper used by data generation scripts."""
    solver = CavitySolver(tau=tau)
    U, V, P = solver.solve(nu, uin_max, t_iter, U_initial, V_initial, P_initial)
    if conn is not None:
        conn.send((U, V, P, None))
        conn.close()
    return U, V, P, None


def sample_on_uniform_grid(gfu, nx=32, ny=32):
    """Legacy helper — kept for plotting utilities."""
    mesh = gfu.components[0].space.mesh
    x_points = np.linspace(0, 1, nx)
    y_points = np.linspace(0, 1, ny)

    vel = gfu.components[0]
    pres = gfu.components[1]
    mpts = [mesh(xi, yi) for yi in y_points for xi in x_points]

    uv = np.array([vel(mp) for mp in mpts])
    pp = np.array([pres(mp) for mp in mpts])

    U = uv[:, 0].reshape(ny, nx)
    V = uv[:, 1].reshape(ny, nx)
    P = pp.reshape(ny, nx)
    return U, V, P


if __name__ == "__main__":
    U = np.zeros((32, 32))
    V = np.zeros((32, 32))
    P = np.zeros((32, 32))
    U, V, P = CavitySolver(tau=0.001).solve(
        nu=0.001, uin_max=1.0, t_iter=250000,
        U_initial=U, V_initial=V, P_initial=P
    )
    print("U range:", U.min(), U.max())