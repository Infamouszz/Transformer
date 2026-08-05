import torch
from typing import Type

from Src.NeuralNetwork.NnModules import Layers
from Src.NeuralNetwork.Settings.Config import DEVICE

class BatchGradientDescentMomentum:
    def __init__(self, batch_size, alpha, beta):
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.V_dW = None
        self.V_dB = None


    def update(self, W, B, total_layers, feedforward, calculate_hits, backpropagation, X_tensor, Y_tensor):
        if self.V_dW is None or self.V_dB is None:
            self.initialize_momentum(W, B)

        batches_limit, global_indexes, epoch_hits = self.initialize_batch_index(X_tensor)

        for start, end in batches_limit:
            batch_indexes = global_indexes[start:end]

            batch_X = X_tensor[batch_indexes]
            batch_Y = Y_tensor[batch_indexes]

            previous_A = batch_X

            A, Z, previous_A = feedforward(total_layers, previous_A, W, B)

            epoch_hits += calculate_hits(previous_A, batch_Y)

            dW_list, dB_list = backpropagation(total_layers, W, B, A, Z, batch_Y, batch_X)

            W, B = self.step(W, B, dW_list, dB_list)

        return self.alpha, epoch_hits, W, B

    def step(self, W, B, dW_list, dB_list):
        for i in range(len(W)):
            self.V_dW[i].mul_(self.beta).add_(dW_list[i], alpha=(1 - self.beta))
            self.V_dB[i].mul_(self.beta).add_(dB_list[i], alpha=(1 - self.beta))

            W[i].sub_(self.alpha * self.V_dW[i])
            B[i].sub_(self.alpha * self.V_dB[i])

        return W, B

    def initialize_batch_index(self, X_tensor):
        total_samples = X_tensor.shape[0]
        batches_limit = [(i, min(i + self.batch_size, total_samples)) for i in range(0, total_samples, self.batch_size)]

        global_indexes = torch.randperm(total_samples, device=DEVICE)
        epoch_hits = torch.tensor(0, device=DEVICE)

        return batches_limit, global_indexes, epoch_hits

    def initialize_momentum(self, W, B):

        self.V_dW = [torch.zeros_like(w) for w in W]
        self.V_dB = [torch.zeros_like(b) for b in B]



