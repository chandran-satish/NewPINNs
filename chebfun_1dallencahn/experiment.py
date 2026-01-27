# experiment.py
import matlab.engine
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time

from model import Net
from initialization import solver_tr, T, pairs, first_time, initialcondition, device
from config import ITERATIONS, ALPHA, lr

class PDEExperiment:
    def __init__(self):
        self.device = device
        self.net = Net().to(self.device)
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), self.lr)
        self.iterations = ITERATIONS
        self.alpha = ALPHA
        self.T = T
        self.initial_cond = initialcondition
        self.batch_size = 32
        self.pairs = pairs
        self.first_time = first_time
        self.solver_tr = solver_tr

    def load_NN(self):
        file_path = "patn\to\pretrained\weights.pth"
        self.net.load_state_dict(torch.load(file_path, map_location=self.device))
        self.net.to(self.device)
        print(f"Model weights loaded from {file_path}")

    def train(self):
        start_time = time.time()
        self.net.train() 
        # Uncomment if you want to use pretrained weights
        # self.load_NN()
        
        for iter in range(self.iterations):
            if (iter != 0) and (iter % 1000 == 0):
                self.save_model(file_path=f"weights_Allen–Cahn_{iter}.pth")
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
          
                solver_out = self.solver_tr.bridge(even_rows_preds, second_column_expanded, self.device)
            
                loss_solver = self.criterion(odd_rows_preds.reshape(-1,1), solver_out.reshape(-1,1))
                
                loss += loss_solver

                first_time_pred = self.net(self.first_time)
                loss_initial = self.criterion(first_time_pred, self.initial_cond)
                loss += self.alpha * loss_initial
                loss = loss / self.batch_size
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (iter + 1) % 5 == 0:
                print(f"Iter [{iter+1}/{self.iterations}], Loss: {loss.item():.10f}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"this is the elapsed time: {elapsed_time}")
        
    def save_model(self, file_path="pde_model_weights.pth"):
        """
        Saves the trained network weights to a file.
        """
        torch.save(self.net.state_dict(), file_path)
        print(f"Model weights saved to {file_path}")