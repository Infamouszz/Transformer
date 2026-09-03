import torch
from Src.Transformer.Settings.Config import DEVICE

class CausalMask:
    def __init__(self):
        pass

    @staticmethod
    def initialize_causal_mask(causal_mask_size):
        causal_mask = torch.full([causal_mask_size,causal_mask_size], -1e9, dtype=torch.float32).to(DEVICE)

        triangle = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0).to(DEVICE)

        return triangle.to(DEVICE)

