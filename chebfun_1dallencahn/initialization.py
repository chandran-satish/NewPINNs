import torch
import numpy as np
from config import T0, T1, DT, N_i, X0, X1, N_POINTS, X0, X1, alpha0, alpha1, Dalpha, input_number
from solver import MatlabPDESolver

print("initializing.....")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Dx = 1.0/(N_POINTS-1)

time_steps = torch.linspace(T0, T1, steps=round((T1 - T0) / DT) + 1, device=device)       # shape (T,)

if N_i == 1:
    time_points = time_steps.clone()
else:
    idx = torch.arange(time_steps.size(0), device=time_steps.device)
    mask = (idx % N_i) == 0
    time_points = time_steps[mask]

T = len(time_points)
xset   = torch.linspace(X0, X1, steps=N_POINTS, device=device) 
alphaset   = torch.linspace(alpha0, alpha1, steps=round((alpha1 - alpha0)/Dalpha) + 1, device=device)
R = len(alphaset)

t_grid, alpha_grid, x_grid= torch.meshgrid(time_points,
                                                 alphaset,
                                                 xset,
                                                 indexing='ij')

inputs = torch.stack((t_grid, alpha_grid, x_grid), dim=-1)

inputs_flat = inputs.view(-1, input_number)
first_time = inputs_flat.view(T,-1, input_number)[0]

total = inputs_flat.view(T * R,-1 , input_number)
a = total[:-R]
b = total[R:]
pairs = torch.stack([a, b], dim=1)

def initial_condition(x):
    return x**2 * torch.cos(np.pi * x)

initialcondition = initial_condition(xset)
initialcondition = initialcondition.unsqueeze(1).repeat(R, 1)

print("initializing matlab engine")
solver_tr = MatlabPDESolver(N_i)
print("matlab engine initialized")