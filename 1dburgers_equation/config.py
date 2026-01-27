# config.py
import numpy as np

# Time parameters
T0 = 0
T1 = 1
DT_train = 0.1
DT_solver = 0.0005
N_i = 200

# Spatial parameters
X0 = -1
X1 = 1
N_POINTS = 101

# Nu parameters
nu0 = 0.01
nu1 = 0.05
Dnu = 0.01

# Training hyperparameters
ITERATIONS = 4000
ALPHA = 1
lr = 1e-3

# NN parameters
input_number = 3
output_number = 1
