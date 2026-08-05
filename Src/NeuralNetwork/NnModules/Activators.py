import torch

class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        return torch.relu(x)

class Softmax:
    def __init__(self, dim = 1):
        self.dim = dim
    def forward(self, x):
        return torch.softmax(x, dim=self.dim)