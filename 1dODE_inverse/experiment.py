# experiment.py
import torch
import torch.optim as optim
import torch.nn as nn

from model import Net, lambda_trainable, k_trainable, x0_trainable
    
from config import ITERATIONS, ALPHA, lr_f, lr_inv
from initilization import ODEsolver, T, time_steps, pairs, first_time, initial_condition, noisy_data, device

class ODEExperiment:
    def __init__(self):
        self.device = device
        self.initial_cond = initial_condition
        self.T = T
        self.noisy_data = noisy_data
        self.pairs = pairs
        self.first_time = first_time
        self.time_steps = time_steps

        self.lambda_trainable = lambda_trainable
        self.k_trainable = k_trainable
        self.x0_trainable = x0_trainable

        self.ODEsolver = ODEsolver
        self.net = Net().to(self.device)
        self.criterion = nn.MSELoss()
        self.lr_f = lr_f
        self.lr_inv = lr_inv
        self.forward_optimizer = optim.Adam(self.net.parameters(), lr_f)
        self.inverse_optimizer = optim.Adam([self.lambda_trainable,
                                            self.k_trainable,
                                            self.x0_trainable], lr_inv)
        
        self.iterations = ITERATIONS
        self.alpha = ALPHA
        self.batch_size = 64

    def load_NN(self):
        file_path = "path/to/the/pretrained/weights.pth"
        self.net.load_state_dict(torch.load(file_path, map_location=self.device))
        self.net.to(self.device)
        print(f"Model weights loaded from {file_path}")

    def train(self):
        self.net.train() 
        # Uncomment this if you aim to train pretrained weights
        # self.load_NN()
        for iter in range(self.iterations):
            perm = torch.randperm(self.pairs.size(0))
            pairs_shuffled = pairs[perm]
            P = pairs_shuffled.size(0)

            batched_pairs = torch.split(pairs_shuffled, self.batch_size, dim=0)
            flattened_batches = [
            pair_batch.contiguous().view(-1, pairs_shuffled.size(-1))  # [#pairs*2, 67]
            for pair_batch in batched_pairs
            ]

            for batch in flattened_batches:
                loss_f = 0
                first_time_pred = self.net(self.first_time)
                batch_pred = self.net(batch)

                even_rows_preds = batch_pred[0::2]
                odd_rows_preds = batch_pred[1::2]

                second_column = batch[:, 1]
               
                second_column_expanded = second_column.unsqueeze(-1)

                third_column = batch[:, 2]
                
                third_column_expanded = third_column.unsqueeze(-1) 

                solver_out = self.ODEsolver.solver(even_rows_preds, second_column_expanded[0::2], third_column_expanded[0::2])

                
                loss_solver = self.criterion(odd_rows_preds, solver_out)
                loss_f += loss_solver

                loss_initial = self.criterion(first_time_pred, self.initial_cond)
                loss_f += loss_initial

                loss_f =  loss_f / self.batch_size

                self.forward_optimizer.zero_grad()
                loss_f.backward()
                self.forward_optimizer.step()
                

            # inverse part
            for p in self.net.parameters():
                p.requires_grad_(False)

            lambda_vec = self.lambda_trainable.view(1,1).expand(self.T, 1)
            k_vec = self.k_trainable.view(1,1).expand(self.T, 1)
            x0_vec = self.x0_trainable.view(1,1).expand(self.T, 1)

            input = torch.cat([self.time_steps.unsqueeze(1), lambda_vec, k_vec, x0_vec], dim=1)

            prediction = self.net(input)

            loss_inv = self.criterion(prediction, self.noisy_data)
            self.inverse_optimizer.zero_grad()
            loss_inv.backward()
            self.inverse_optimizer.step()

            for p in self.net.parameters():
                p.requires_grad_(True)

            if (iter + 1) % 20 == 0:
                print(f"Iter [{iter+1}/{self.iterations}]--> Loss_forward: {loss_f.item():.10f}, Loss_inverse: {loss_inv.item():.10f}")
                print(f"This is the updated lambda: {self.lambda_trainable.item()}")
                print(f"This is the updated k: {self.k_trainable.item()}")
                print(f"This is the updated x0: {self.x0_trainable.item()}")
                print("================================================================")


    def save_model(self, file_path="pde_model_weights.pth"):
        """
        Saves the trained network weights to a file.
        """
        torch.save(self.net.state_dict(), file_path)
        print(f"Model weights saved to {file_path}")
