# solver.py
import numpy as np
import torch

class ODESolver:
    def __init__(self, DT, N_i, device):
        self.DT = np.array(DT)
        self.device = device
        self.N_i = N_i

    def solver(self, input_value, second_column, third_column):
        
        if hasattr(input_value, 'cpu'):
            input_np = input_value.detach().cpu().numpy()
        else:
            input_np = np.array(input_value)

        if hasattr(second_column, 'cpu'):
            second_column = second_column.detach().cpu().numpy()
        else:
            second_column = np.array(second_column)

        if hasattr(third_column, 'cpu'):
            third_column = third_column.detach().cpu().numpy()
        else:
            third_column = np.array(third_column)
        
        if input_np.ndim == 1:
            input_np = input_np[:, None]
        
        results = {}
        results[0] = input_np
        for i in range(1, self.N_i + 1):
            results[i] = results[i-1] - ((self.DT / self.N_i) * results[i-1] * second_column) + (third_column * (self.DT / self.N_i))

        results_tensor = torch.tensor(results[self.N_i], dtype=torch.float32, device=self.device)
        return results_tensor
