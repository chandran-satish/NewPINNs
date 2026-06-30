import os
import sys
import random
from contextlib import contextmanager

import numpy as np
import h5py

from utils_sm_ngsolve import Elasticity3DSolver


# ----------------------------------------------------------------------
# Globals: these define the problem instance used throughout the project.
# Keep in sync with config_sm.yaml.
# ----------------------------------------------------------------------
# Material / loading
#
# Note on scaling: linear elasticity is exactly scale-invariant -- multiplying
# the traction by k scales displacement by k. The network has Kaiming init,
# so its natural output scale is O(1). We pick the force so that true tip
# displacement is also O(1):
#     δ_tip ≈ traction · L / E = 0.3 · 3.0 / 1.0 = 0.9.
# An earlier draft used FORCE_VEC = (1e-3, 0, 0), which gave δ ≈ 3e-3 and
# left the network unable to learn the O(1e-3) scale from a O(1) init.
E_FIXED      = 1.0                      # Young's modulus (fixed)
FORCE_VEC    = (0.3, 0.0, 0.0)          # tip traction on the "force" face
NU_MIN       = 0.10                     # ν_poisson sweep range
NU_MAX       = 0.45                     # avoid the incompressible limit

# Geometry / mesh
MAXH         = 0.10                     # FE element size
FE_ORDER     = 2                        # element order
N_HOLES      = 3
CHAMFER      = 0.03
BOX_DIMS     = (3.0, 0.6, 1.0)

# Sampling grid (nz, ny, nx) -- matches beam aspect ratio
NZ, NY, NX   = 32, 16, 96

# Solver framing
N_ITER_GT    = 500                        # ground-truth: direct CG
TAU          = 1.0                      # Richardson step (not used for GT)


# ----------------------------------------------------------------------
@contextmanager
def utils_suppress_stdout():
    """Silence chatty NGSolve output during data generation."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# ----------------------------------------------------------------------
def utils_sample_single():
    """
    Return a single training sample of the conditioning array.

    Returns
    -------
    nu_norm_arr : ndarray (nz, ny, nx) float
        Constant-valued array, all entries = ν_norm ∈ [0, 1].
        The full-array representation mirrors how the cavity stores the
        Reynolds-number image; the network reads it as a channel.
    """
    nu_norm = random.random()  # uniform on [0, 1]
    return nu_norm * np.ones((NZ, NY, NX), dtype=np.float32)


def _unnormalize_nu(nu_norm):
    """Map ν_norm ∈ [0, 1] back to physical ν_poisson ∈ [NU_MIN, NU_MAX]."""
    return (NU_MAX - NU_MIN) * float(nu_norm) + NU_MIN


# ----------------------------------------------------------------------
def utils_save_train_dataset(get_data_func, num_samples, save_path, myseed):
    """
    Save a training dataset of conditioning arrays only.

    For NewPINNs training, the solver-consistency loop is run on-the-fly
    during training (the network's prediction is fed back into the solver
    at each step), so the training file does NOT store ground-truth
    displacement fields. The mask is stored once globally because it's a
    fixed property of the geometry.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    random.seed(myseed)
    np.random.seed(myseed)

    # Build a solver instance once to get the geometry mask.
    print("Building solver to extract geometry mask ...", flush=True)
    with utils_suppress_stdout():
        solver = Elasticity3DSolver(
            maxh=MAXH, order=FE_ORDER, n_holes=N_HOLES,
            box_dims=BOX_DIMS, chamfer=CHAMFER,
            nx=NX, ny=NY, nz=NZ,
        )
    mask = solver.mask.astype(np.float32)
    print(f"Mask coverage: {mask.mean():.3f}", flush=True)

    with h5py.File(save_path, "w") as f:
        f.create_dataset("mask", data=mask)
        for i in range(num_samples):
            img1 = get_data_func().astype(np.float32)
            if i == 0:
                f.create_dataset(
                    "img1", (num_samples, *img1.shape), dtype=img1.dtype
                )
            f["img1"][i] = img1
            if (i + 1) % 50 == 0 or i == num_samples - 1:
                print(f"  saved {i + 1}/{num_samples}", flush=True)


def utils_save_valid_dataset(get_data_func, num_samples, save_path, myseed):
    """
    Save a validation dataset with conditioning arrays AND ground-truth
    displacement fields (direct CG solve, n_iter=0).
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    random.seed(myseed)
    np.random.seed(myseed)

    print("Building persistent solver for validation data ...", flush=True)
    with utils_suppress_stdout():
        solver = Elasticity3DSolver(
            maxh=MAXH, order=FE_ORDER, n_holes=N_HOLES,
            box_dims=BOX_DIMS, chamfer=CHAMFER,
            nx=NX, ny=NY, nz=NZ,
        )
    mask = solver.mask.astype(np.float32)
    print(f"Mask coverage: {mask.mean():.3f}", flush=True)

    shape3d = (NZ, NY, NX)
    zeros3d = np.zeros(shape3d, dtype=np.float32)

    with h5py.File(save_path, "w") as f:
        f.create_dataset("mask", data=mask)

        for i in range(num_samples):
            img1 = get_data_func().astype(np.float32)
            nu_poisson = _unnormalize_nu(img1[0, 0, 0])
            print(f"Sample {i}: ν = {nu_poisson:.4f}", flush=True)

            # Direct CG ground truth from a zero IC.
            with utils_suppress_stdout():
                Ueq, Veq, Weq, _ = solver.solve(
                    E=E_FIXED,
                    nu_poisson=nu_poisson,
                    force_vec=FORCE_VEC,
                    n_iter=N_ITER_GT,
                    U_initial=zeros3d,
                    V_initial=zeros3d,
                    W_initial=zeros3d,
                )

            if i == 0:
                f.create_dataset(
                    "img1", (num_samples, *img1.shape), dtype=img1.dtype
                )
                f.create_dataset(
                    "Ueq", (num_samples, *shape3d), dtype=np.float32
                )
                f.create_dataset(
                    "Veq", (num_samples, *shape3d), dtype=np.float32
                )
                f.create_dataset(
                    "Weq", (num_samples, *shape3d), dtype=np.float32
                )

            f["img1"][i] = img1
            f["Ueq"][i] = Ueq.astype(np.float32)
            f["Veq"][i] = Veq.astype(np.float32)
            f["Weq"][i] = Weq.astype(np.float32)

            if (i + 1) % 10 == 0 or i == num_samples - 1:
                print(f"  saved {i + 1}/{num_samples}", flush=True)


# ----------------------------------------------------------------------
def main():
    """
    Convention (mirrors utils_cavity_data.py):
      seed 42 for training data, seed 43 for validation data.
    Edit the calls below to control which files are (re)generated.
    """

    # Training set: conditioning only (no ground truth)
    num_train = 256
    utils_save_train_dataset(
        utils_sample_single,
        num_samples=num_train,
        save_path=f"./data/train_data_{num_train}.h5",
        myseed=42,
    )

    # Validation set: conditioning + ground-truth fields
    num_valid = 32
    utils_save_valid_dataset(
        utils_sample_single,
        num_samples=num_valid,
        save_path=f"./data/valid_data_{num_valid}.h5",
        myseed=43,
    )


if __name__ == "__main__":
    main()