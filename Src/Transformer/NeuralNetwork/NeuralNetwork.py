import torch
import torch.nn.functional as F
from Src.Transformer.Settings.Config import DEVICE


class NeuralNetwork:
    def __init__(self, d_model, d_ffn):
        self.d_model = d_model
        self.d_ffn = d_ffn

        self.W1 = torch.randn(d_model, d_ffn, device = DEVICE) * (2.0 / d_model)**0.5
        self.b1 = torch.zeros(d_ffn, device=DEVICE)

        self.W2 = torch.randn(d_ffn, d_model, device=DEVICE) * (2.0 / d_model) ** 0.5
        self.b2 = torch.zeros(d_model, device=DEVICE)

        self.dW1, self.db1 = None, None
        self.dW2, self.db2 = None, None

    def gelu_tanh_derivative(self, Z):

        u = 0.7978845608 * (Z + 0.044715 * torch.pow(Z, 3))
        tanh_u = torch.tanh(u)
        du_dZ = 0.7978845608 * (1.0 + 0.134145 * torch.pow(Z, 2))

        d_gelu = 0.5 * (1.0 + tanh_u) + 0.5 * Z * (1.0 - torch.pow(tanh_u, 2)) * du_dZ
        return d_gelu

    def forward(self, X):
        self.X = X
        self.Z1 = torch.matmul(X, self.W1) + self.b1
        self.A1 = F.gelu(self.Z1, approximate ='tanh')
        self.Z2 = torch.matmul(self.A1, self.W2) + self.b2

        return self.Z2, self.A1

    def backward(self, dL_dZ2):
        shape_original = dL_dZ2.shape

        dL_dZ2_flat = dL_dZ2.reshape(-1, dL_dZ2.shape[-1])
        A1_flat = self.A1.reshape(-1, self.A1.shape[-1])
        X_flat = self.X.reshape(-1, self.X.shape[-1])

        self.dW2 = torch.matmul(A1_flat.T, dL_dZ2_flat)
        self.db2 = dL_dZ2_flat.sum(dim=0)

        dL_dA1_flat = torch.matmul(dL_dZ2_flat, self.W2.T)

        d_gelu_flat = self.gelu_tanh_derivative(self.Z1.reshape(-1, self.Z1.shape[-1]))
        dL_dZ1_flat = dL_dA1_flat * d_gelu_flat

        self.dW1 = torch.matmul(X_flat.T, dL_dZ1_flat)
        self.db1 = dL_dZ1_flat.sum(dim=0)

        dL_dX_flat = torch.matmul(dL_dZ1_flat, self.W1.T)
        dL_dX = dL_dX_flat.reshape(shape_original)

        return dL_dX