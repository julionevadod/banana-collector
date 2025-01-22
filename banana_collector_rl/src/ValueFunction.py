import torch
from torch import nn

class ValueFunction(nn.Module):

    def __init__(self, input_size: int, output_size: int, fc1: int = 64, fc2: int = 64):
        super(ValueFunction, self).__init__()
        self.fc1 = nn.Linear(input_size,fc1,dtype=torch.float64)
        self.fc2 = nn.Linear(fc1,fc2,dtype=torch.float64)
        self.output = nn.Linear(fc2,output_size,dtype=torch.float64)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(nn.functional.relu(x))
        x = self.output(nn.functional.relu(x))
        return x
