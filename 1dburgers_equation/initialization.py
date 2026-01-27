# initialization.py
import torch
import numpy as np
from config import T0, T1, DT_train, DT_solver, N_i, X0, X1, N_POINTS, X0, X1, nu0, nu1, Dnu
from solver import BurgersSolver

print("initializing.....")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Dx = 1.0/(N_POINTS-1)

time_steps = torch.linspace(T0, T1, steps=round((T1 - T0) / DT_train) + 1, device=device)       # shape (T,)

T = len(time_steps)
xset   = torch.linspace(X0, X1, steps=N_POINTS, device=device) 
nuset   = torch.linspace(nu0, nu1, steps=round((nu1 - nu0)/Dnu) + 1, device=device)

R = len(nuset)

t_grid, nu_grid, x_grid= torch.meshgrid(time_steps,
                                        nuset,
                                        xset,
                                        indexing='ij')

inputs = torch.stack((t_grid, nu_grid, x_grid), dim=-1)

inputs_flat = inputs.view(-1, 3)
first_time = inputs_flat.view(T,-1, 3)[0]

total = inputs_flat.view(T * R,-1 , 3)
a = total[:-R]
b = total[R:]
pairs = torch.stack([a, b], dim=1)

def initial_condition(x):
    return -torch.sin(np.pi * x)

initialcondition = initial_condition(xset)
initialcondition = initialcondition.unsqueeze(1).repeat(R, 1)

print("initializing the solver")
solver = BurgersSolver(DT_solver, Dx, N_i)
print("solver initialized")