import torch
from Src.Transformer.Settings.Config import DEVICE


class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.eps = eps
        self.d_model = d_model
        self.gamma = torch.ones(d_model).to(DEVICE)
        self.beta = torch.zeros(d_model).to(DEVICE)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std_inv = 1.0 / torch.sqrt(var + self.eps)

        x_hat = (x - mean) * std_inv

        self.x_hat = x_hat
        self.std_inv = std_inv

        return self.gamma * x_hat + self.beta

    def backward(self, dY):
        sum_dims = tuple(range(dY.ndim - 1))
        d_gamma = (dY * self.x_hat).sum(dim=sum_dims)
        d_beta = dY.sum(dim=sum_dims)

        d_hat = dY * self.gamma

        D = self.d_model
        sum_d_hat = d_hat.sum(dim=-1, keepdim=True)
        sum_d_hat_xhat = (d_hat * self.x_hat).sum(dim=-1, keepdim=True)

        dx = (1.0 / D) * self.std_inv * (
                D * d_hat - sum_d_hat - self.x_hat * sum_d_hat_xhat
        )

        return dx, d_gamma, d_beta