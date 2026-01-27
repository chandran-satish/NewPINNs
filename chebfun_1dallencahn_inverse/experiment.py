# experiment.py
import matlab.engine
import torch
import torch.optim as optim
import torch.nn as nn
import time

from model import Net, alpha_trainable
from initialization import solver_tr, T, pairs, first_time, initialcondition, device, noisy_data, xset, real_t
from config import ITERATIONS, ALPHA, lr_f, lr_inv

class PDEExperiment:
    def __init__(self):
        self.device = device
        self.solver_tr = solver_tr
        self.net = Net().to(self.device)
        self.lr_f = lr_f
        self.lr_inv = lr_inv
        self.alpha_trainable = alpha_trainable
        self.forward_optimizer = optim.Adam(self.net.parameters(), lr_f)
        self.inverse_optimizer = optim.Adam([self.alpha_trainable], lr_inv)
        self.criterion = nn.MSELoss()
        
        self.iterations = ITERATIONS
        self.alpha = ALPHA
        self.T = T
        self.real_t = real_t
        self.xset = xset
        self.noisy_data = noisy_data

        self.initial_cond = initialcondition
        self.batch_size = 32
        self.pairs = pairs
        self.first_time = first_time

    def load_NN(self):
        file_path = "path/to/the/pretrained/weights.pth"
        self.net.load_state_dict(torch.load(file_path, map_location=self.device))
        self.net.to(self.device)
        print(f"Model weights loaded from {file_path}")

    def train(self):
        start_time = time.time()
        self.net.train() 
        # uncomment this if you aim to use pretrained weights for training
        # self.load_NN()
        
        print("training begins")
        for iter in range(self.iterations):
            if (iter != 0) and (iter % 3000 == 0):
                self.save_model(file_path=f"weights_Allen–Cahn_inverse_{iter}.pth")
         
            perm = torch.randperm(self.pairs.size(0))
            pairs_shuffled = pairs[perm]
            P = pairs_shuffled.size(0)
       
            batched_pairs = torch.split(pairs_shuffled, self.batch_size, dim=0)

            
            for batch in batched_pairs:
                loss_f = 0
                
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
                loss_f += loss_solver

                first_time_pred = self.net(self.first_time)
                loss_initial = self.criterion(first_time_pred, self.initial_cond)
                loss_f += loss_initial

                loss_f = loss_f / self.batch_size

                self.forward_optimizer.zero_grad()
                loss_f.backward()
                self.forward_optimizer.step()

            for p in self.net.parameters():
                p.requires_grad_(False)

            t_grid, alpha_grid, x_grid= torch.meshgrid(self.real_t, self.alpha_trainable.unsqueeze(0), self.xset, indexing='ij')
            inputs = torch.stack((t_grid, alpha_grid, x_grid), dim=-1).view(-1,3)

            prediction = self.net(inputs)

            loss_inv = self.criterion(prediction, self.noisy_data)
            self.inverse_optimizer.zero_grad()
            loss_inv.backward()
            self.inverse_optimizer.step()

            for p in self.net.parameters():
                p.requires_grad_(True)
            
            if (iter + 1) % 5 == 0:
                print(f"Iter [{iter+1}/{self.iterations}], Loss Forward: {loss_f.item():.10f}, Loss_inverse: {loss_inv.item():.10f}")
                print(f"This is the updated alpha: {self.alpha_trainable.item()}")
                print("================================================================")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"this is the elapsed time: {elapsed_time}")


    def save_model(self, file_path="pde_model_weights.pth"):
        """
        Saves the trained network weights to a file.
        """
        torch.save(self.net.state_dict(), file_path)
        print(f"Model weights saved to {file_path}")