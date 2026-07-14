import math
import torch
import torch_directml
from tqdm import tqdm
import h5py
class NeuralNetwork:
    def __init__(self, h5_file_path):
        self.file_path = h5_file_path

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using GPU")
        elif torch_directml.is_available():
            self.device = torch_directml.device()
            print("Using XPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")


    def train(self, hidden_layers, alpha, beta, epochs, batch_size):
        with h5py.File(self.file_path, 'r') as h5f:
            X_tensor = torch.tensor(h5f['X'][:], dtype=torch.float32).to(self.device)
            Y_tensor = torch.tensor(h5f['Y'][:], dtype=torch.float32).to(self.device)

        total_samples = X_tensor.shape[0]
        batchs_limit = [(i, min(i + batch_size, total_samples)) for i in range(0, total_samples, batch_size)]

        X_shape = X_tensor.shape[1]
        Y_shape = Y_tensor.shape[1]

        W, B, V_dW, V_dB, total_layers = self.W_B_initializer(X_shape, Y_shape, hidden_layers)

        A = [None] * total_layers
        Z = [None] * total_layers
        dZ = [None] * total_layers

        with torch.no_grad():
            for epoch in tqdm(range(epochs), desc="Training model..."):
                global_indexes = torch.randperm(total_samples, device = self.device)
                epoch_hits = torch.tensor(0, device = self.device)

                for start, end in batchs_limit:
                    batch_indexes = global_indexes[start:end]

                    batch_X = X_tensor[batch_indexes]
                    batch_Y = Y_tensor[batch_indexes]

                    batch_m = batch_X.shape[0]
                    previous_A = batch_X

                    for layer in range(total_layers):
                        is_out = (layer == total_layers - 1)
                        A_temp, Z_temp = self.layer_calculator(previous_A, W[layer], B[layer], is_out)
                        A[layer] = A_temp
                        Z[layer] = Z_temp
                        previous_A = A_temp

                    epoch_hits += self.calculate_hits(previous_A, batch_Y)

                    for layer_upd in range(total_layers - 1, -1, -1):
                        if layer_upd == total_layers - 1:
                            dZ_temp = self.update_out_layer_parameters(W[layer_upd], B[layer_upd], V_dW[layer_upd],
                                                                  V_dB[layer_upd], A[layer_upd - 1], A[layer_upd], batch_Y,
                                                                  batch_m, alpha, beta)
                            dZ[layer_upd] = dZ_temp
                        else:
                            input_A = batch_X if layer_upd == 0 else A[layer_upd - 1]
                            dZ_temp = self.update_hidden_layer_parameters(W[layer_upd], W[layer_upd + 1], B[layer_upd],
                                                                     V_dW[layer_upd], V_dB[layer_upd], Z[layer_upd],
                                                                     dZ[layer_upd + 1], input_A, alpha, beta)
                            dZ[layer_upd] = dZ_temp

                if epoch % 100 == 0:
                    accuracy = (epoch_hits.item() / total_samples) * 100
                    print(f"Epoch: {epoch} | Accuracy: {accuracy:.2f}% | Learning rate: {alpha}")

        return W, B


    def W_B_initializer(self, X_shape, Y_shape, hidden_layers):
        W_initialized = []
        B_initialized = []
        V_dW = []
        V_dB = []
        architecture = [X_shape] + hidden_layers + [Y_shape]

        for layer in range(1, len(architecture)):
            w = torch.randn(architecture[layer - 1], architecture[layer], device=self.device, dtype=torch.float32) * math.sqrt(
                2.0 / architecture[layer - 1])
            b = torch.zeros((1, architecture[layer]), device=self.device, dtype=torch.float32)

            W_initialized.append(w)
            B_initialized.append(b)

            V_dW.append(torch.zeros_like(w))
            V_dB.append(torch.zeros_like(b))

        total_layers = len(architecture) - 1
        return W_initialized, B_initialized, V_dW, V_dB, total_layers


    def update_out_layer_parameters(self, W, B, V_dW, V_dB, A_lower, A, Y, m, alpha, beta):
        dZ, dW, dB = self.output_derivative_calculator(A, A_lower, Y, m)

        V_dW.mul_(beta).add_(dW)
        V_dB.mul_(beta).add_(dB)

        W.sub_(alpha * V_dW)
        B.sub_(alpha * V_dB)
        return dZ


    def update_hidden_layer_parameters(self, W, W_upper, B, V_dW, V_dB, Z, dZ_upper, A_lower, alpha, beta):
        dZ, dW, dB = self.hidden_derivative_calculator(dZ_upper, W_upper, Z, A_lower)

        V_dW.mul_(beta).add_(dW)
        V_dB.mul_(beta).add_(dB)

        W.sub_(alpha * V_dW)
        B.sub_(alpha * V_dB)
        return dZ


    def layer_calculator(self, A, W, B, is_Out_Put):
        z_new_result = torch.matmul(A, W) + B

        if is_Out_Put:
            z_shifted = z_new_result - torch.max(z_new_result, dim=1, keepdim=True)[0]
            a_new_result = torch.softmax(z_shifted, dim=1)
        else:
            a_new_result = torch.relu(z_new_result)

        return a_new_result, z_new_result


    def output_derivative_calculator(self, Y_final, A_lower, y, m):
        dZo = (Y_final - y) / m
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