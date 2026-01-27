import torch
import numpy as np
from config import T0, T1, DT, N_i, lambda_1, lambda_2, D_lambda, x01, x02, Dx0, k1, k2, Dk, real_lambda, real_k, real_x0
from solver import ODESolver

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

time_steps = torch.linspace(T0, T1, steps=round((T1 - T0) / DT) + 1, device=device)  
T = len(time_steps)
lambda_set = torch.linspace(lambda_1, lambda_2, steps=round((lambda_2 - lambda_1)/D_lambda) + 1, device=device) 
x0set   = torch.linspace(x01, x02, steps=round((x02 - x01)/Dx0) + 1, device=device)
kset   = torch.linspace(k1, k2, steps=round((k2 - k1)/Dk) + 1, device=device)


analytical_data =(real_k / real_lambda) + ((real_x0 - (real_k / real_lambda)) * torch.exp(-real_lambda * time_steps.view(T,1)))
y_exact = analytical_data.cpu().numpy().flatten() 
noise=0.0
np.random.seed(0)
noise = y_exact * noise * np.random.randn(len(y_exact))
y_noisy_np = y_exact + noise
noisy_data = torch.from_numpy(y_noisy_np).to(device=device, dtype=torch.float32).view_as(analytical_data)

T = len(time_steps)
N = len(lambda_set)
M = len(x0set)
K = len(kset)

t_grid, lambda_grid, k_grid, x0_grid = torch.meshgrid(time_steps,
                                                 lambda_set,
                                                 kset,
                                                 x0set,
                                                 indexing='ij')

inputs = torch.stack((t_grid, lambda_grid, k_grid, x0_grid), dim=-1) 

inputs_flat = inputs.view(-1, 4)

num_batches = 64
reshaped_input = inputs_flat.view(T,-1,4)
first_time = reshaped_input[0]

total = reshaped_input
a = total[:-1]
b = total[1:]
pairs = torch.stack([a, b], dim=2)

pairs = pairs.view(-1, 2, reshaped_input.size(-1))

initial_condition = x0set.unsqueeze(0).repeat(N * K, 1).reshape(-1, 1)

ODEsolver = ODESolver(DT, N_i ,device)