from torch import nn

class ValueFunction(nn.Module):
    def __init__(self, input_size: int, fc1: int, fc2: int, output_size: int):
        self.fc1 = nn.Linear(input_size,fc1)
        self.fc2 = nn.Linear(fc1,fc2)
        self.output = nn.Linear(fc2, output_size)

    def __forward__(self, x):
        x = self.fc1(input)
        x = self.fc2(nn.ReLU()(x))
        x = self.output(nn.ReLU()(x))
        return x
