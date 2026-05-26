import math
from importlib import resources
import numpy as np
import pickle
import torch
import torch_directml
from triton.language import float32

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
elif torch_directml.is_available():
    device = torch_directml.device()
    print("Using XPU")
else:
    device = torch.device("cpu")
    print("Using CPU")



with resources.files('ModelsAi').joinpath('matrix_gen_5.pkl').open('rb') as m:
    loaded_matrix = pickle.load(m)

with resources.files('ModelsAi').joinpath('matrix_y_gen_5.pkl').open('rb') as my:
    loaded_matrix_y = pickle.load(my)

X_entered = loaded_matrix['X']
Y_entered = loaded_matrix_y['Y']


X = np.array(X_entered, dtype=np.float32)
X = X / np.max(X)

array_list = list(Y_entered.values())
Y = np.vstack(array_list)
Y = np.array(Y, dtype=np.float32)

Y = torch.tensor(Y, dtype=torch.float32).to(device)
X = torch.tensor(X, dtype=torch.float32).to(device)

print("Class distribuition: ", torch.sum(Y, dim = 0).cpu())

def train(hidden_layers, alpha, epochs, batch_size):
    m = X.shape[1] * 2
    A = {}
    Z = {}
    W, B, total_layers = W_B_initializer(hidden_layers)
    dZ = {}
    for epoch in range(epochs):
        shuffle = torch.randperm(X.shape[0])
        shuffled_Y = Y[shuffle,:]
        shuffled_X = X[shuffle,:]

        for i in range(0, X.shape[0], batch_size):
            batch_X = shuffled_X[i : i + batch_size, :]
            batch_Y = shuffled_Y[i : i + batch_size, :]
            batch_m = batch_X.shape[0]


            previous_A = batch_X
            for layer in range(total_layers):
                is_out = is_Output_Layer(layer, total_layers)

                A_temp, Z_temp = layer_calculator(previous_A, W[layer], B[layer], is_out)
                A[layer] = A_temp
                Z[layer] = Z_temp

                previous_A = A_temp

            for layer_upd in range(total_layers - 1, -1, -1):
                if layer_upd == total_layers - 1:
                    W_temp,B_temp,dZ_temp = update_out_layer_parameters(W[layer_upd], B[layer_upd], A[layer_upd - 1], A[layer_upd], batch_Y, batch_m, alpha)
                    W[layer_upd] = W_temp
                    B[layer_upd] = B_temp
                    dZ[layer_upd] = dZ_temp
                else:
                    input_A = batch_X if layer_upd == 0 else A[layer_upd - 1]
                    W_temp,B_temp,dZ_temp = update_hidden_layer_parameters(W[layer_upd],W[layer_upd+1], B[layer_upd], Z[layer_upd], dZ[layer_upd + 1], input_A, alpha)
                    W[layer_upd] = W_temp
                    B[layer_upd] = B_temp
                    dZ[layer_upd] = dZ_temp
        if epoch % 400 == 0:
            accuracy = accuracy_calculator(previous_A, batch_Y)
            print("Epoch: ", epoch, "Accuracy: ", accuracy)
    return W, B

def W_B_initializer(hidden_layers):
    W_initialized = {}
    B_initialized = {}
    architecture = build_architecture(X,Y, hidden_layers)
    layer = 1
    index = 0
    while layer <= len(architecture) - 1:
        W_initialized[index] = torch.randn(architecture[layer - 1], architecture[layer]).to(device) * math.sqrt(2 / architecture[layer - 1])
        B_initialized[index] = torch.zeros((1, architecture[layer])).to(device)
        layer += 1
        index += 1
    total_layers = len(architecture) - 1
    return W_initialized, B_initialized, total_layers

def build_architecture(X, Y, hidden_layers):
    architecture = {}
    architecture[0] = X.shape[1]

    indexer = 1

    while indexer <= len(hidden_layers):
        architecture[indexer] = hidden_layers[indexer-1]
        indexer += 1
    architecture[indexer] = Y.shape[1]
    return architecture

def update_out_layer_parameters(W, B, A_lower, A, Y , m, alpha):
    dZ, dW, dB = output_derivative_calculator(A, A_lower, Y, m)
    W.sub_(alpha * dW)
    B.sub_(alpha * dB)

    return W, B, dZ

def update_hidden_layer_parameters(W, W_upper, B, Z, dZ_upper, A_lower, alpha):
    dZ, dW, dB = hidden_derivative_calculator(dZ_upper, W_upper, Z, A_lower)
    W.sub_(alpha * dW)
    B.sub_(alpha * dB)

    return W, B, dZ

def softmax(Z):
    Z_clamp = torch.clamp(Z, min = -80.0, max = 80.0)

    Z_shifted = Z_clamp - torch.max(Z_clamp, dim = 1, keepdim=True).values

    exp_Z = torch.exp(Z_shifted)
    probabilities = exp_Z / torch.sum(exp_Z, dim = 1, keepdim = True)
    return probabilities

def layer_calculator(A, W, B, is_Out_Put):
    z_new_result = torch.matmul(A, W) + B
    if is_Out_Put:
        a_new_result = softmax(z_new_result)
    else:
        z_new_result.clamp_(min = -500.0, max = 500.0)
        a_new_result = ReLU(z_new_result)

    return a_new_result, z_new_result

def output_derivative_calculator(Y_final, A_lower, y, m):
    dAo = (Y_final - y) / m
    dZo = dAo
    dWo = torch.matmul(A_lower.T, dZo)
    dBo = torch.sum(dZo, dim=0, keepdim=True)

    dWo.clamp_(min = -1.0, max = 1.0)
    dBo.clamp_(min = -1.0,max = 1.0)

    return dZo, dWo, dBo

def hidden_derivative_calculator(dZ_upper, W, Z, A):
    dAh = torch.matmul(dZ_upper, W.T)
    dZh = dAh * (Z > 0).float()
    dWh = torch.matmul(A.T, dZh)
    dBh = torch.sum(dZh, dim=0, keepdim=True)

    dWh.clamp_(min = -1.0, max = 1.0)
    dBh.clamp_(min = -1.0, max = 1.0)

    return dZh, dWh, dBh
def accuracy_calculator(A_output, Y):
    predictions = torch.argmax(A_output, dim=1)

    targets = torch.argmax(Y, dim=1)

    hits = torch.sum(predictions == targets).item()

    accuracy = (hits / len(Y)) * 100

    return accuracy

def ReLU(value):
    result = torch.relu(value)
    return result

def is_Output_Layer(layer, total_layer):
    if layer == total_layer - 1:
        return True
    else:
        return False


def predict(X, W, B):
    A_new = X

    layers = len(W)

    for i in range(layers):
        Z = torch.matmul(A_new, W[i]) + B[i]

        if i < layers - 1:
            A_new = torch.relu(Z)
        else:
            A_new = Z

    return A_new

W_new, B_new = train([128, 64, 32, 16], 0.001, 15000, 1024)

parameters = {
    'W': W_new,
    'B': B_new,
}

path = resources.files('ModelsAi').joinpath('parameters_gen_7.pkl')
with path.open('wb') as p:
    pickle.dump(parameters, p)
    print("Parameters saved successfully")




