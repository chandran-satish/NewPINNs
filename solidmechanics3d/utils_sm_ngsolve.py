import os
import numpy as np

from netgen.occ import Box, Cylinder, Y, X
from ngsolve import (
    Mesh, VectorH1, GridFunction, BilinearForm, LinearForm,
    CoefficientFunction, InnerProduct, Sym, Grad, Trace, Id,
    dx, ds, TaskManager, SetNumThreads,
)
from ngsolve import preconditioners, solvers

# VoxelCoefficient: top-level in current NGSolve, ngsolve.internal in older builds.
try:
    from ngsolve import VoxelCoefficient
except ImportError:
    from ngsolve.internal import VoxelCoefficient


class Elasticity3DSolver:
    """
    Persistent NGSolve solver for 3D linear elasticity on a box with
    cylindrical holes, set up for the NewPINNs solver-consistency loss.

    Structure (build once, solve many) is taken from the cavity solver:
    geometry, mesh, FE space, sampling grid and in-domain mask are built
    once in __init__; the stiffness matrix + BDDC preconditioner are
    cached on (E, nu_poisson) and only rebuilt when that pair changes.

    Iterative method (the "optimizer")
    ----------------------------------
    Preconditioned conjugate gradient with BDDC preconditioning -- the
    same solver the elasticity3D.ipynb notebook uses
    (``solvers.CGSolver(a.mat, pre, tol=1e-8)``). Capped at ``n_iter``
    iterations to give the partial-solver-correction semantics required
    by the NewPINNs operator T(u). The IC u_0 (passed in as
    U_initial, V_initial, W_initial on the uniform grid) is honoured
    via the residual reformulation: instead of solving K u = f from
    scratch, we solve

        K * delta = f - K * u_0      with CG, maxsteps = n_iter,

    then return u = u_0 + delta.  This is mathematically equivalent to
    "CG with initial guess u_0" and corresponds exactly to running
    n_iter CG iterations of the original system from the IC.

    Geometry (matches the notebook)
    -------------------------------
    Box (Lx, Ly, Lz) minus n_holes cylindrical holes along Y at
    x = 0.5, 1.5, ... (radius 0.25, length 0.8). Chamfered at the
    box/cylinder intersections. Face x = 0   named "fix"   (Dirichlet).
                                   Face x = Lx  named "force" (Neumann).

    Linear elasticity (matches the notebook)
    ----------------------------------------
        eps(u)  = sym(grad u)
        sigma   = 2*mu*eps + lambda*tr(eps)*I
        a(u,v)  = int sigma(eps(u)) : eps(v)  dx
        L(v)    = int_{force} t . v  ds
    with Lame parameters derived from (E, nu_poisson).
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        maxh=0.1,
        order=3,
        n_holes=3,
        box_dims=(3.0, 0.6, 1.0),
        chamfer=0.03,
        nx=96, ny=20, nz=32,
        num_threads=None,
    ):
        # ---- threading ----
        if num_threads is None:
            num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
        SetNumThreads(num_threads)

        # ---- geometry (built once) ----
        Lx, Ly, Lz = box_dims
        box = Box((0, 0, 0), (Lx, Ly, Lz))
        box.faces.name = "outer"
        cyl = sum(
            [Cylinder((0.5 + i, 0, 0.5), Y, 0.25, 0.8) for i in range(n_holes)]
        )
        cyl.faces.name = "cyl"
        geo = box - cyl

        if chamfer is not None and chamfer > 0:
            cylboxedges = geo.faces["outer"].edges * geo.faces["cyl"].edges
            cylboxedges.name = "cylbox"
            geo = geo.MakeChamfer(cylboxedges, chamfer)

        geo.faces.Min(X).name = "fix"
        geo.faces.Max(X).name = "force"

        ngmesh = geo.GenerateMesh(maxh=maxh)
        try:
            self.mesh = Mesh(ngmesh).Curve(3)
        except TypeError:
            self.mesh = ngmesh.Curve(3)

        self.box_dims = (Lx, Ly, Lz)
        self.order = int(order)

        # ---- FE space (built once) ----
        self.fes = VectorH1(self.mesh, order=order, dirichlet="fix")
        self.u_trial, self.v_test = self.fes.TnT()
        self.gfu = GridFunction(self.fes)
        self.gfu_int = GridFunction(self.fes)

        # ---- uniform sampling grid + domain mask (built once) ----
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        x_pts = np.linspace(0.0, Lx, self.nx)
        y_pts = np.linspace(0.0, Ly, self.ny)
        z_pts = np.linspace(0.0, Lz, self.nz)

        self._mesh_pts = []
        self._mask = np.zeros((self.nz, self.ny, self.nx), dtype=bool)
        # Note: self.mesh(xi, yi, zi) returns a MeshPoint regardless of
        # containment; only gfu(mp) raises for out-of-domain points.
        for kz, zi in enumerate(z_pts):
            for jy, yi in enumerate(y_pts):
                for ix, xi in enumerate(x_pts):
                    try:
                        mp = self.mesh(xi, yi, zi)
                        _ = self.gfu(mp)
                    except Exception:
                        continue
                    self._mesh_pts.append((kz, jy, ix, mp))
                    self._mask[kz, jy, ix] = True
        self._grid_shape = (self.nz, self.ny, self.nx)

        # ---- assembly cache ----
        self._cached_key = None
        self._a = None
        self._pre = None

    # ------------------------------------------------------------------
    @property
    def mask(self):
        return self._mask

    # ------------------------------------------------------------------
    def _assemble(self, E, nu_poisson):
        key = (float(E), float(nu_poisson))
        if key == self._cached_key:
            return

        mu = E / 2.0 / (1.0 + nu_poisson)
        lam = E * nu_poisson / ((1.0 + nu_poisson) * (1.0 - 2.0 * nu_poisson))

        def stress(eps):
            return 2.0 * mu * eps + lam * Trace(eps) * Id(3)

        u, v = self.u_trial, self.v_test
        eps_u = Sym(Grad(u))
        eps_v = Sym(Grad(v))

        with TaskManager():
            self._a = BilinearForm(
                InnerProduct(stress(eps_u), eps_v).Compile() * dx
            )
            # BDDC must be registered before Assemble.
            self._pre = preconditioners.BDDC(self._a)
            self._a.Assemble()

        self._cached_key = key

    # ------------------------------------------------------------------
    def solve(
        self,
        E,
        nu_poisson,
        force_vec,
        n_iter,
        U_initial,
        V_initial,
        W_initial,
    ):
        """
        NewPINNs solver operator.

        Parameters
        ----------
        E : float
            Young's modulus.
        nu_poisson : float
            Poisson's ratio.
        force_vec : sequence of 3 floats
            Surface traction on the "force" face (x = Lx).
        n_iter : int
            Number of CG iterations to apply, starting from the IC.
            n_iter == 0 is a special case: CG to tol = 1e-8 (the
            notebook's behaviour, used for ground-truth generation).
            n_iter > 0 gives the partial-correction operator T(u).
        U_initial, V_initial, W_initial : ndarray (nz, ny, nx)
            IC displacement components on the uniform grid. Values
            outside the body are ignored.

        Returns
        -------
        U, V, W : ndarray (nz, ny, nx)
            Displacement components after n_iter CG iterations from
            the IC (entries outside the body are 0).
        mask : ndarray (nz, ny, nx) of bool
        """
        self._assemble(E, nu_poisson)

        # ---- traction load (reassembled per call -- cheap) ----
        force_cf = CoefficientFunction(tuple(float(c) for c in force_vec))
        f = LinearForm(force_cf * self.v_test * ds("force"))
        with TaskManager():
            f.Assemble()

        # ---- inject IC into the FE space via VoxelCoefficient ----
        # VoxelCoefficient takes (corner_min, corner_max, array, linear=True);
        # for 3D it expects array indexed as (z, y, x).
        U64 = np.float64(U_initial)
        V64 = np.float64(V_initial)
        W64 = np.float64(W_initial)
        Lx, Ly, Lz = self.box_dims
        ufunc = VoxelCoefficient((0, 0, 0), (Lx, Ly, Lz), U64, linear=True)
        vfunc = VoxelCoefficient((0, 0, 0), (Lx, Ly, Lz), V64, linear=True)
        wfunc = VoxelCoefficient((0, 0, 0), (Lx, Ly, Lz), W64, linear=True)
        self.gfu_int.Set(CoefficientFunction((ufunc, vfunc, wfunc)))

        # Enforce homogeneous Dirichlet on "fix" by zeroing the full
        # vector and copying only free (interior) DOFs from gfu_int.
        self.gfu.vec[:] = 0.0
        free_dofs = self.fes.FreeDofs()
        for dof in range(self.fes.ndof):
            if free_dofs[dof]:
                self.gfu.vec[dof] = self.gfu_int.vec[dof]

        # ---- iterate using CG with BDDC (same optimizer as the notebook) ----
        with TaskManager():
            if n_iter == 0:
                # Notebook behaviour: CG to tolerance from zero IC,
                # used for ground-truth data generation.
                inv = solvers.CGSolver(
                    self._a.mat, self._pre, plotrates=False, tol=1e-8
                )
                self.gfu.vec.data = inv * f.vec
            else:
                # NewPINNs operator T(u_0): n_iter CG iterations starting
                # from u_0. Implemented via the residual reformulation
                #     K * delta = f - K * u_0
                # solved with CG capped at maxsteps = n_iter, then
                # u <- u_0 + delta.  This is mathematically the same as
                # running CG on the original system with initial guess
                # u_0 for n_iter iterations.
                res = f.vec.CreateVector()
                res.data = f.vec - self._a.mat * self.gfu.vec
                try:
                    inv = solvers.CGSolver(
                        self._a.mat, self._pre,
                        plotrates=False, maxiter=int(n_iter),
                    )
                except TypeError:
                    # Older NGSolve uses 'maxsteps' instead of 'maxiter'.
                    inv = solvers.CGSolver(
                        self._a.mat, self._pre,
                        plotrates=False, maxsteps=int(n_iter),
                    )
                delta = inv * res
                self.gfu.vec.data += delta

        U, V, W = self._sample()
        return U, V, W, self._mask.copy()

    # ------------------------------------------------------------------
    def _sample(self):
        U = np.zeros(self._grid_shape, dtype=np.float64)
        V = np.zeros(self._grid_shape, dtype=np.float64)
        W = np.zeros(self._grid_shape, dtype=np.float64)
        gfu = self.gfu
        for (kz, jy, ix, mp) in self._mesh_pts:
            d = gfu(mp)
            U[kz, jy, ix] = d[0]
            V[kz, jy, ix] = d[1]
            W[kz, jy, ix] = d[2]
        return U, V, W


# ======================================================================
# Convenience wrappers (backward-compatible style with the cavity module)
# ======================================================================

def run_and_capture_ngsolve_elasticity(
    E=1.0,
    nu_poisson=0.3,
    force_vec=(0.3, 0.0, 0.0),
    n_iter=20,
    U_initial=None, V_initial=None, W_initial=None,
    tag="",
):
    global _MODULE_SOLVER
    if "_MODULE_SOLVER" not in globals() or _MODULE_SOLVER is None:
        _MODULE_SOLVER = Elasticity3DSolver()
    return _MODULE_SOLVER.solve(
        E, nu_poisson, force_vec, n_iter,
        U_initial, V_initial, W_initial,
    )


def run_ngsolve_elasticity_custom(
    E=1.0,
    nu_poisson=0.3,
    force_vec=(0.3, 0.0, 0.0),
    n_iter=20,
    U_initial=None, V_initial=None, W_initial=None,
    conn=None, tag="",
):
    solver = Elasticity3DSolver()
    out = solver.solve(E, nu_poisson, force_vec, n_iter,
                       U_initial, V_initial, W_initial)
    if conn is not None:
        conn.send(out)
        conn.close()
    return out


# ======================================================================
if __name__ == "__main__":
    solver = Elasticity3DSolver(maxh=0.15, order=2, nx=48, ny=12, nz=16)
    shape = (solver.nz, solver.ny, solver.nx)
    zero = np.zeros(shape)

    # Ground truth: notebook-equivalent (full CG to tolerance).
    U, V, W, mask = solver.solve(
        E=1.0, nu_poisson=0.3, force_vec=(0.3, 0.0, 0.0),
        n_iter=0,
        U_initial=zero, V_initial=zero, W_initial=zero,
    )
    print(f"[CG-to-tol]  mask coverage: {mask.mean():.3f}")
    print(f"             U range: {U[mask].min():.4e} {U[mask].max():.4e}")
    print(f"             V range: {V[mask].min():.4e} {V[mask].max():.4e}")
    print(f"             W range: {W[mask].min():.4e} {W[mask].max():.4e}")

    # Partial-correction operator: a few CG steps from a zero IC.
    Up, Vp, Wp, _ = solver.solve(
        E=1.0, nu_poisson=0.3, force_vec=(0.3, 0.0, 0.0),
        n_iter=5,
        U_initial=zero, V_initial=zero, W_initial=zero,
    )
    print(f"[CG, n_iter=5] U range: {Up[mask].min():.4e} {Up[mask].max():.4e}")
    print(f"               relative ||u_5 - u_true|| / ||u_true||: "
          f"{np.linalg.norm((Up - U)[mask]) / max(np.linalg.norm(U[mask]), 1e-12):.4e}")