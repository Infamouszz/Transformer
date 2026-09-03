import torch
from Src.Transformer.Settings.Config import DEVICE

class WeightInitializer:
    def __init__(self, d_model):
        self.d_model = d_model

    def init_weights_HE(self):
        std = torch.sqrt(torch.tensor(2.0 / self.d_model))

        Wq = torch.randn(self.d_model, self.d_model, device=DEVICE, dtype=torch.float32) * std
        Wk = torch.randn(self.d_model, self.d_model, device=DEVICE, dtype=torch.float32) * std
        Wv = torch.randn(self.d_model, self.d_model, device=DEVICE, dtype=torch.float32) * std
        Wo = torch.randn(self.d_model, self.d_model, device=DEVICE, dtype=torch.float32) * std

        return Wq, Wk, Wv, Wo
