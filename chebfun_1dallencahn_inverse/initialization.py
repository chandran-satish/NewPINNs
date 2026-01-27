import torch
import numpy as np
from config import T0, T1, DT, N_i, X0, X1, N_POINTS, X0, X1, alpha0, alpha1, Dalpha, input_number
from solver import MatlabPDESolver

print("initializing.....")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Dx = 1.0/(N_POINTS-1)

time_steps = torch.linspace(T0, T1, steps=round((T1 - T0) / DT) + 1, device=device) 

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

# data generation for inverse model
t0_data = 0
t1_data = 0.7
dt_data = 0.05

time_steps_data = torch.linspace(t0_data, t1_data, steps=round((t1_data - t0_data) / dt_data) + 1, device=device)
T_data = len(time_steps_data)

true_solutions = []
current = initial_condition(xset.unsqueeze(1).to(device))
true_solutions.append(current.cpu().numpy())

for step in range(1, T_data):
    current_np = current.detach().cpu().numpy()
    next_np = solver_tr.solver_test(current_np)
    current = torch.from_numpy(next_np).to(device=device, dtype=torch.float32)
    true_solutions.append(current)

real_data = true_solutions[-1]
noise_std = 0.1 * real_data.abs().mean()  
noise = torch.randn_like(real_data) * noise_std
noisy_data = real_data + noise

real_t = 0.7
real_t = torch.tensor([real_t], dtype = torch.float32, device=device)
print("initialization completed")