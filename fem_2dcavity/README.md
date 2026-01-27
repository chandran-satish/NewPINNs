# NewPINNs Lid-Driven Cavity (NGSolve + UNet)

This repository implements a **solver-coupled Physics-Informing Neural Network (NewPINNs)** for the 2D **lid-driven cavity flow** problem.  
A convolutional neural network (UNet) predicts velocity and pressure fields, which are then **advanced by a finite-element Navier–Stokes solver (NGSolve)** inside the training loop. The network is trained by minimizing **solver-consistency**, rather than PDE residuals.

This codebase supports:
- Dataset generation using **NGSolve**
- UNet-based prediction of velocity and pressure fields
- Online FEM correction during training
- Robust handling of solver NaN/Inf failures
- Distributed training with PyTorch Lightning (DDP)

---

## Problem Setup

We consider the 2D incompressible lid-driven cavity problem on the unit square:
- No-slip walls on left, right, and bottom
- Moving lid with prescribed horizontal velocity on the top boundary
- Reynolds number varies per sample and is provided as input

The neural network predicts:
- Horizontal velocity u_x
- Vertical velocity u_y
- Pressure p

on a **32 × 32 uniform grid**, which is then mapped to the FEM mesh via voxel interpolation.

---

## Repository Structure

```
.
├── pinn_cavity.py          # Main training entry point (PyTorch Lightning)
├── config_cavity.yaml      # YAML configuration for data, model, and training
├── utils_cavity_ngsolve.py # FEM solver + NN → FEM coupling (NGSolve)
├── utils_cavity_unet.py    # UNet architecture + LightningModule
├── utils_cavity_data.py   # Dataset generation and HDF5 utilities
├── data/
│   ├── train_data_*.h5
│   └── valid_data_*.h5
└── README.md
```

---

## Key Components

### FEM Solver (NGSolve)

File: `utils_cavity_ngsolve.py`

- Solves incompressible Navier–Stokes using Taylor–Hood elements (P3–P2)
- Semi-implicit time stepping
- Accepts initial conditions from the neural network
- Runs inside a separate process to avoid PyTorch–NGSolve conflicts

---

### Neural Network Model

File: `utils_cavity_unet.py`

- UNet based on `diffusers.UNet2DModel`
- Input: Reynolds-number image
- Output channels: u_x, u_y, p
- Configurable weight initialization

---

### Solver-Coupled Training (NewPINNs)

File: `pinn_cavity.py`

Training loop:
1. UNet predicts (u_x, u_y, p)
2. Prediction is passed to NGSolve
3. FEM solver advances the solution
4. Loss enforces solver-consistency

Loss:
L = λ_u_x ||u_x − u_x*||² + λ_u_y ||u_y − u_y*||² + λ_p ||p − p*||²

---

### Dataset Generation

File: `utils_cavity_data.py`

- Generates HDF5 datasets
- Each sample contains:
  - Reynolds number field
  - FEM equilibrium solution

---

## Installation

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- PyTorch Lightning
- diffusers
- h5py, yaml, numpy
- NGSolve + Netgen

Install NGSolve:
```
pip install ngsolve
```

---

## Usage

### Generate Datasets
```
python utils_cavity_data.py
```

### Train Model
```
python pinn_cavity.py --config config_cavity.yaml
```
