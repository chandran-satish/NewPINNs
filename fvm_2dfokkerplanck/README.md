# NewPINNs Fokker–Planck Equation (FiPy + UNet)

This repository implements a **solver-coupled Physics-Informing Neural Network (NewPINNs)** for the **2D Fokker–Planck equation** using a finite-volume solver (FiPy).  
A convolutional neural network (UNet) predicts a probability density field, which is then **advanced by a FiPy-based Fokker–Planck solver inside the training loop**. The network is trained by enforcing **solver-consistency**, rather than minimizing PDE residuals.

This setup mirrors the NewPINNs philosophy used in the cavity-flow codebase, but targets **gradient-flow dynamics and equilibrium densities**.

---

## Problem Setup

We consider the Fokker–Planck equation
\[
\partial_t p = \nabla \cdot (D \nabla p + p \nabla V),
\]
on the unit square \([0,1]^2\), where:
- \(p(x,y,t)\) is a probability density,
- \(V(x,y) = \alpha \sin(2\pi x) \sin(2\pi y)\) is a potential,
- \(D = 1\) is the diffusion coefficient.

The equilibrium solution satisfies
\[
p_{\mathrm{eq}} \propto \exp(-V/D),
\]
which is used as ground truth for validation.

The neural network predicts **density fields on a 32×32 grid**, which are then refined by the FiPy solver.

---

## Repository Structure

```text
.
├── pinn_fpe.py              # Main training entry point (PyTorch Lightning)
├── config_fpe.yaml          # YAML configuration for data, model, and training
├── utils_fpe_fipy.py        # FiPy-based Fokker–Planck solver
├── utils_fpe_unet.py        # UNet architecture and LightningModule
├── utils_fpe_data.py        # Dataset generation and HDF5 utilities
├── utils_fpe_inference.py   # Post-training inference and visualization
├── utils_fpe_fipy_test.py   # Standalone FiPy verification scripts
├── data/
│   ├── train_data_*.h5
│   └── valid_data_*.h5
└── README.md
```

---

## Key Components

### Finite-Volume Solver (FiPy)

**File:** `utils_fpe_fipy.py`

- Uses `FiPy` to solve the Fokker–Planck equation
- Discretization:
  - DiffusionTerm
  - ExponentialConvectionTerm (drift)
- Accepts NN-predicted density as an initial condition
- Advances the solution for a fixed number of time steps
- Renormalizes density to preserve probability mass

Core function:
- `run_fipy_custom(...)`

---

### Neural Network Model

**File:** `utils_fpe_unet.py`

- UNet based on `diffusers.UNet2DModel`
- Input: scalar potential amplitude \(\alpha\) encoded as a constant image
- Output: predicted density field \(p(x,y)\)
- Output constrained via sigmoid to enforce positivity
- Configurable weight initialization

---

### Solver-Coupled Training (NewPINNs)

**File:** `pinn_fpe.py`

Training loop:
1. UNet predicts an initial density field
2. Prediction is passed to FiPy
3. FiPy advances the density toward equilibrium
4. Loss enforces consistency between NN output and solver-evolved density

Loss:
```text
L = || p_θ − S(p_θ) ||²
```
where \(S\) denotes the FiPy solver operator.

Robustness features:
- Automatic detection of NaN / Inf solver outputs
- Per-epoch logging of failed samples
- Training continues even if some samples fail

---

### Dataset Generation

**File:** `utils_fpe_data.py`

- Generates HDF5 datasets
- Training data:
  - Input: potential amplitude \(\alpha\)
- Validation data:
  - Input: \(\alpha\)
  - Target: analytic equilibrium density \(\exp(-V/D)\)

Datasets contain:
- `img1`: normalized \(\alpha\) field
- `p_eq`: equilibrium density (validation only)

---

## Installation

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- PyTorch Lightning
- diffusers
- FiPy
- h5py, yaml, numpy, matplotlib

Install FiPy:
```bash
pip install fipy
```

---

## Usage

### Generate Datasets
```bash
python utils_fpe_data.py
```

### Train the Model
```bash
python pinn_fpe.py --config config_fpe.yaml
```

Supports:
- Multi-GPU (DDP)
- Mixed precision
- Checkpointing
- TensorBoard logging

---

## Inference & Visualization

Use:
```bash
python utils_fpe_inference.py
```

This script:
- Loads a trained UNet
- Predicts the equilibrium density
- Compares NN output to analytic solution
- Produces contour and error plots
