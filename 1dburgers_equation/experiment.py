# experiment.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from model import Net
from initialization import solver, T, pairs, first_time, initialcondition, device
from config import ITERATIONS, ALPHA, lr

class PDEExperiment:
    def __init__(self):
        self.device = device
        # Only initialize the solver if requested.
        self.solver = solver
        # Build the neural network and move it to the selected device
        self.net = Net().to(self.device)
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), self.lr)
        
        # Training hyperparameters from config
        self.iterations = ITERATIONS
        self.alpha = ALPHA
        self.T = T
        self.initial_cond = initialcondition
        self.batch_size = 20
        self.pairs = pairs
        self.first_time = first_time

    def load_NN(self):
        file_path = "weights_burgers_3000.pth"
        self.net.load_state_dict(torch.load(file_path, map_location=self.device))
        self.net.to(self.device)
        print(f"Model weights loaded from {file_path}")

    def train(self):
        self.net.train()
        # uncomment this if you want to use pretrained weights for training
        # self.load_NN()
        
        print("training begins")
        for iter in range(self.iterations):
            if (iter != 0) and (iter % 200 == 0):
                self.save_model(file_path=f"weights_burgers_{30000+iter}.pth")
          
            perm = torch.randperm(self.pairs.size(0))
            pairs_shuffled = pairs[perm]
            P = pairs_shuffled.size(0)
    
            batched_pairs = torch.split(pairs_shuffled, self.batch_size, dim=0)
            
            for batch in batched_pairs:
                loss = 0
                batch_shape0 = batch.shape[0]
                batch_shape2 = batch.shape[2]

                flattened_batch = batch.view(-1, 3)
                
                batch_pred = self.net(flattened_batch)
                
                even_rows_preds = batch_pred.view(2 * batch_shape0, batch_shape2, -1)[0::2]
                odd_rows_preds = batch_pred.view(2 * batch_shape0, batch_shape2, -1)[1::2]

                second_column = batch[:, :, :, 1]
                
                second_column_expanded = second_column.unsqueeze(-1)
                second_column_expanded = second_column_expanded.view(2 * batch_shape0, batch_shape2, -1)[0::2]
                second_column_expanded = second_column_expanded[:, 0, :]

                solver_out = self.solver(even_rows_preds.squeeze(-1), second_column_expanded)

                loss_solver = self.criterion(odd_rows_preds.reshape(-1,1), solver_out.reshape(-1,1))
                loss += loss_solver
            
                first_time_pred = self.net(self.first_time)

                loss_initial = self.criterion(first_time_pred, self.initial_cond)

                loss += loss_initial
                loss = loss / batch_shape0

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (iter + 1) % 25 == 0:
                print(f"Iter [{iter+1}/{self.iterations}], Loss: {loss.item():.10f}")

    def save_model(self, file_path="pde_model_weights.pth"):
        """
        Saves the trained network weights to a file.
        """
        torch.save(self.net.state_dict(), file_path)
        print(f"Model weights saved to {file_path}")