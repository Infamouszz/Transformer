import math
import torch
from Src.Transformer.Settings.Config import DEVICE


class Unembedding:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def initialize_unembedding_HE(self, d_model):
        W_unembed = torch.randn(d_model, self.vocab_size, device=DEVICE) * (2.0 / d_model) ** 0.5
        b_unembed = torch.zeros(self.vocab_size, device=DEVICE)

        return W_unembed, b_unembed

    def forward(self, final_X, W_unembed, b_unembed):


        logits = torch.matmul(final_X, W_unembed) + b_unembed
        return logits

    def backward(self, dZ, final_X_norm, W_unembedding):
        shape_original = final_X_norm.shape

        dZ_flat = dZ.reshape(-1, dZ.shape[-1])
        X_flat = final_X_norm.reshape(-1, final_X_norm.shape[-1])

        dW = torch.matmul(X_flat.T, dZ_flat)

        db = dZ_flat.sum(dim=0)

        dX_flat = torch.matmul(dZ_flat, W_unembedding.T)
        dX_norm_final = dX_flat.reshape(shape_original)

        return dX_norm_final, dW, db



