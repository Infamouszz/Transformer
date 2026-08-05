import torch
import math
from Src.NeuralNetwork.Settings.Config import DEVICE

class HeInitialization:
    def __init__(self):
        pass

    def initialize(self, X_shape, Y_shape, hidden_layers):
        W_initialized = []
        B_initialized = []
        architecture = [X_shape] + hidden_layers + [Y_shape]

        for layer in range(1, len(architecture)):
            w, b = self.he_calculator(architecture, layer)

            W_initialized.append(w)
            B_initialized.append(b)


        total_layers = len(architecture) - 1
        return W_initialized, B_initialized, total_layers

    def he_calculator(self, architecture, layer):
        w = torch.randn(architecture[layer - 1], architecture[layer], device=DEVICE, dtype=torch.float32) * math.sqrt(2.0 / architecture[layer - 1])
        b = torch.zeros((1, architecture[layer]), device=DEVICE, dtype=torch.float32)

        return w, b