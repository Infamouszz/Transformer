import h5py
import torch
from NeuralNetwork import NeuralNetwork
from Src.NeuralNetwork import NnModules
from Src.NeuralNetwork.Settings.Config import DEVICE

with h5py.File("C:/Users/Guilherme/PycharmProjects/DashboardAi/Src/NeuralNetwork/meus_dados_prontos.h5", 'r') as h5f:
    Xz_tensor = torch.tensor(h5f['X'][:], dtype=torch.float32).to(DEVICE)
    Yz_tensor = torch.tensor(h5f['Y'][:], dtype=torch.float32).to(DEVICE)
nm = NnModules
initializer = nm.Initializers.HeInitialization()
hidden_activation = nm.Activators.ReLU()
output_activation = nm.Activators.Softmax()
losses = nm.Losses.CrossEntropyLoss()
layers = nm.Layers.layers([16,8,4])
optimizer = nm.Optimizers.BatchGradientDescentMomentum(
    16384,
    0.001,
    0.9,
)
nn = NeuralNetwork(
    Xz_tensor,
    Yz_tensor,
    initializer,
    hidden_activation,
    output_activation,
    losses,
    optimizer,
    layers
)
nn.train(15000)
