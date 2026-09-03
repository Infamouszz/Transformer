import torch
from Src.Transformer.Settings.Config import DEVICE


class PaddingMask:
    def __init__(self):
        pass

    def initialize_padding_mask(self, score):
        padding_mask = score[score == 10256]
        padding_mask[padding_mask == True] = -1e9
        padding_mask[padding_mask == False] = 0
        padding_mask.unsqueeze(0).unsqueeze(0).to(DEVICE)

        return padding_mask