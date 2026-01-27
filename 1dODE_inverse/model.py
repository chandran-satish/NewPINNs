# model.py
import torch
import torch.nn as nn
from config import input_number, output_number, initial_lambda, initial_k, initial_x0

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_number, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, output_number)
        self.tanh = nn.Tanh()

    def forward(self, inp):
        x = self.tanh(self.fc1(inp))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.tanh(self.fc4(x))
        x = self.tanh(self.fc5(x))
        x = self.fc6(x)
        return x
    
lambda_trainable = nn.Parameter(torch.tensor(initial_lambda, dtype=torch.float32))
k_trainable = nn.Parameter(torch.tensor(initial_k, dtype=torch.float32))
x0_trainable = nn.Parameter(torch.tensor(initial_x0, dtype=torch.float32))
