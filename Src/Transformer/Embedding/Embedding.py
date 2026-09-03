import math
import torch
from Src.Transformer.Settings.Config import DEVICE

class Embedding:
    def __init__(self, d_model):
        self.d_model = d_model
    
    def initialize_embedding_HE(self,vocab_size):
        fan_in = vocab_size
        fan_out = self.d_model
    
        std = torch.sqrt(torch.tensor(2.0 / fan_in))
    
        embedding = torch.randn(fan_in, fan_out, device=DEVICE, dtype=torch.float32) * std
    
        return embedding

    def initialize_sinusoidal_positional_embedding(self, max_len):
        pe = torch.zeros(max_len, self.d_model, device=DEVICE, dtype=torch.float32)
    
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
    
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
    
        return pe
    
    def forward(self,  tokens, embedding, positional_embedding):
        if isinstance(tokens, torch.Tensor):
            tokens_tensor = tokens.clone().detach().to(dtype=torch.long, device=DEVICE)
        else:
            tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
        tokens_embedding = embedding[tokens_tensor]
    
        return tokens_embedding + positional_embedding

