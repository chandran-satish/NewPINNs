# config.py
import torch
import numpy as np

# NN parameters
input_number = 4
output_number = 1

# problem parameters
initial_lambda = 0.1
initial_k = -2.9
initial_x0 = 0

# initial condition
real_lambda = 0.85
real_k = -0.65
real_x0 = 2.15

# Time parameters
T0 = 0
T1 = 4
DT = 0.1
N_i = 10

# problem parameters
lambda_1 = 0.1
lambda_2 = 1
D_lambda = 0.3

# initial condition
k1 = -3
k2 = 0
Dk = 0.5

# initial condition
x01 = 0
x02 = 3
Dx0 = 0.5

# Training hyperparameters
ITERATIONS = 20000
ALPHA = 1
lr_f = 1e-3
lr_inv = 1e-3

