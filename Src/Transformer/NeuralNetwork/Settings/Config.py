import torch
import torch_directml

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using GPU")
elif torch_directml.is_available():
    DEVICE = torch_directml.device()
    print("Using XPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")