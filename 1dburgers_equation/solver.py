# solver.py
import torch
import numpy as np
from config import T0, X0, X1, N_POINTS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BurgersSolver:
    def __init__(self, dt, dx, N_i):
        self.N_i = N_i
        self.dt = dt
        self.dx = dx
        self.device = device

    def __call__(self, input_value, nu):
        
        if hasattr(input_value, 'cpu'):
            input_value = input_value.detach().cpu().numpy()
        else:
            input_value = np.array(input_value, dtype=float)

        if hasattr(nu, 'cpu'):
            nu = nu.detach().cpu().numpy()
        else:
            nu = np.array(nu, dtype=float)

        for _ in range(self.N_i):
            un = input_value.copy()
            # interior points: vectorized over all cases
            input_value[:, 1:-1] = (
                un[:, 1:-1]
                - un[:, 1:-1] * self.dt/(2*self.dx) * (un[:, 2:] - un[:, :-2])
                + nu * self.dt/self.dx**2 * (un[:, 2:] - 2*un[:, 1:-1] + un[:, :-2])
            )
            # boundary conditions (for all cases)
            input_value[:,  0] = 0
            input_value[:, -1] = 0
        
        output_tensor = torch.from_numpy(input_value).to(device=device, dtype=torch.float32)

        return output_tensor
