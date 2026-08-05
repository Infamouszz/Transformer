import torch
from tqdm import tqdm
from NnModules import Initializers
from NnModules import Activators
from NnModules import Losses
from typing import Type
from Src.NeuralNetwork.NnModules import Optimizers, Layers


class NeuralNetwork:
    def __init__(self, X_tensor, Y_tensor, initializer: Type[Initializers], hidden_activator: Type[Activators],output_activator: Type[Activators],
                 losses: Type[Losses], optimizer: Type[Optimizers], layers: Type[Layers]):
        self.X_tensor = X_tensor
        self.Y_tensor = Y_tensor

        self.layers = layers
        self.initializer = initializer
        self.hidden_activator = hidden_activator
        self.output_activator = output_activator
        self.loss = losses
        self.optimizer = optimizer

    def train(self, epochs):

        total_samples = self.X_tensor.shape[0]

        X_shape = self.X_tensor.shape[1]
        Y_shape = self.Y_tensor.shape[1]

        W, B, total_layers = self.initializer.initialize(X_shape, Y_shape, self.layers.hidden_layers)

        with torch.no_grad():
            for epoch in tqdm(range(epochs), desc="Training model..."):

                alpha, epoch_hits, W, B = self.optimizer.update(
                    W,
                    B,
                    total_layers,
                    self.feedforward,
                    self.calculate_hits,
                    self.backpropagation,
                    self.X_tensor,
                    self.Y_tensor,
                )
                if epoch % 100 == 0:
                    accuracy = (epoch_hits.item() / total_samples) * 100
                    print(f"Epoch: {epoch} | Accuracy: {accuracy:.2f}% | Learning rate: {alpha}")

        return W, B

    def feedforward(self, total_layers, previous_A, W, B):
        A = [None] * total_layers
        Z = [None] * total_layers
        for layer in range(total_layers):
            is_out = (layer == total_layers - 1)
            A_temp, Z_temp = self.layer_calculator(previous_A, W[layer], B[layer], is_out)
            A[layer] = A_temp
            Z[layer] = Z_temp
            previous_A = A_temp
        return A, Z, previous_A


    def backpropagation(self, total_layers, W, B, A, Z, batch_Y, batch_X):
        dZ_upper = None
        dW_list = [None] * total_layers
        dB_list = [None] * total_layers

        for layer_upd in range(total_layers - 1, -1, -1):
            if layer_upd == 0:
                A_lower = batch_X
            else:
                A_lower = A[layer_upd - 1]

            if layer_upd == total_layers - 1:
                dZ_upper, dW, dB = self.calculate_out_layer_derivative(
                    A_lower, A[layer_upd], batch_Y
                )
            else:
                dZ_upper, dW, dB = self.calculate_hidden_layer_derivative(
                    W[layer_upd + 1], Z[layer_upd], dZ_upper, A_lower
                )

            dW_list[layer_upd] = dW
            dB_list[layer_upd] = dB
        return dW_list, dB_list

    def calculate_out_layer_derivative(self, A_lower, A, Y):
        dZ, dW, dB = self.output_derivative_calculator(A, A_lower, Y)
        return dZ, dW, dB


    def calculate_hidden_layer_derivative(self, W_upper, Z, dZ_upper, A_lower):
        dZ, dW, dB = self.hidden_derivative_calculator(dZ_upper, W_upper, Z, A_lower)
        return dZ, dW, dB


    def layer_calculator(self, A, W, B, is_Out_Put):
        z_new_result = torch.matmul(A, W) + B

        if is_Out_Put:
            z_shifted = z_new_result - torch.max(z_new_result, dim=1, keepdim=True)[0]
            a_new_result = self.output_activator.forward(z_shifted)
        else:
            a_new_result = self.hidden_activator.forward(z_new_result)

        return a_new_result, z_new_result


    def output_derivative_calculator(self, Y_final, A_lower, y):
        dZo = self.loss.derivative(Y_final, y)
        dWo = torch.matmul(A_lower.T, dZo)
        dBo = torch.sum(dZo, dim=0, keepdim=True)

        return dZo, dWo, dBo


    def hidden_derivative_calculator(self, dZ_upper, W, Z, A):
        dAh = torch.matmul(dZ_upper, W.T)
        dZh = dAh * (Z > 0)
        dWh = torch.matmul(A.T, dZh)
        dBh = torch.sum(dZh, dim=0, keepdim=True)

        return dZh, dWh, dBh


    def calculate_hits(self, A_output, Y):
        predictions = torch.argmax(A_output, dim=1)
        targets = torch.argmax(Y, dim=1)
        return torch.sum(predictions == targets)


    def predict(self, X, W, B):
        A_new = X
        layers = len(W)

        for i in range(layers):
            Z = torch.matmul(A_new, W[i]) + B[i]
            if i < layers - 1:
                A_new = torch.relu(Z)
            else:
                A_new = torch.softmax(Z, dim=1)

        return A_new