import random
from fileinput import close
from operator import index
from platform import architecture
import numpy as np
import pickle

with open(r'C:\Users\guilh\PycharmProjects\DashboardAi\Src\NeuralNetwork\ModelsAi\matrix_gen_5.pkl', 'rb') as m:
    loaded_matrix = pickle.load(m)

with open(r'C:\Users\guilh\PycharmProjects\DashboardAi\Src\NeuralNetwork\ModelsAi\matrix_y_gen_5.pkl', 'rb') as my:
    loaded_matrix_y = pickle.load(my)

X_entered = loaded_matrix['X']
Y_entered = loaded_matrix_y['Y']


X = np.array(X_entered, dtype=np.float32)
X = X / np.max(X)

array_list = list(Y_entered.values())
Y = np.vstack(array_list)
Y = np.array(Y, dtype=np.float32)

print("X: ",X.shape)
print("Y: ",Y.shape)

def train(hidden_layers, alpha, epochs):
    m = X.shape[1] * 2
    A = {}
    Z = {}
    W, B, total_layers = W_B_initializer(hidden_layers)
    dZ = {}
    for epoch in range(epochs):
        previous_A = X
        for layer in range(total_layers):
            is_out = is_Output_Layer(layer, total_layers)

            A_temp, Z_temp = layer_calculator(previous_A, W[layer], B[layer], is_out)
            A[layer] = A_temp
            Z[layer] = Z_temp

            previous_A = A_temp

        for layer_upd in range(total_layers - 1, -1, -1):
            if layer_upd == total_layers - 1:
                W_temp,B_temp,dZ_temp = update_out_layer_parameters(W[layer_upd], B[layer_upd], A[layer_upd - 1], A[layer_upd], m, alpha)
                W[layer_upd] = W_temp
                B[layer_upd] = B_temp
                dZ[layer_upd] = dZ_temp
            else:
                input_A = X if layer_upd == 0 else A[layer_upd - 1]
                W_temp,B_temp,dZ_temp = update_hidden_layer_parameters(W[layer_upd],W[layer_upd+1], B[layer_upd], Z[layer_upd], dZ[layer_upd + 1], input_A, alpha)
                W[layer_upd] = W_temp
                B[layer_upd] = B_temp
                dZ[layer_upd] = dZ_temp
        if epoch % 500 == 0:
            print("Epoch: ", epoch)

    return W, B

def W_B_initializer(hidden_layers):
    W_initialized = {}
    B_initialized = {}
    architecture = build_architecture(X,Y, hidden_layers)
    layer = 1
    index = 0
    while layer <= len(architecture) - 1:
        W_initialized[index] = np.random.randn(architecture[layer - 1], architecture[layer]) * np.sqrt(2 / architecture[layer - 1])
        B_initialized[index] = np.zeros((1, architecture[layer]))
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

def update_out_layer_parameters(W, B, A_lower, A, m, alpha):
    dZ, dW, dB = output_derivative_calculator(A, A_lower, m)
    W_updated = W - alpha * dW
    B_updated = B - alpha * dB

    return W_updated, B_updated, dZ

def update_hidden_layer_parameters(W, W_upper, B, Z, dZ_upper, A_lower, alpha):
    dZ, dW, dB = hidden_derivative_calculator(dZ_upper, W_upper, Z, A_lower)
    W_updated = W - alpha * dW
    B_updated = B - alpha * dB

    return W_updated, B_updated, dZ

def layer_calculator(A, W, B, is_Out_Put):
    z_new_result = np.dot(A, W) + B
    if is_Out_Put:
        a_new_result = z_new_result
    else:
        a_new_result = ReLU(z_new_result)

    return a_new_result, z_new_result

def output_derivative_calculator(Y_final, A_lower, m):
    dAo = (Y_final - Y) / m
    dZo = dAo
    dWo = np.dot(np.transpose(A_lower), dZo)
    dBo = np.sum(dZo, axis=0, keepdims=True)
    dWo = np.clip(dWo, -1.0, 1.0)
    dBo = np.clip(dBo, -1.0, 1.0)

    return dZo, dWo, dBo

def hidden_derivative_calculator(dZ_upper, W, Z, A):
    dAh = np.dot(dZ_upper, np.transpose(W))
    dZh = dAh * (Z > 0)
    dWh = np.dot(np.transpose((X if A is None else A)), dZh)
    dBh = np.sum(dZh, axis=0, keepdims=True)

    return dZh, dWh, dBh


def ReLU(value):
    result = np.maximum(0, value)
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
        Z = np.dot(A_new, W[i]) + B[i]

        if i < layers - 1:
            A_new = np.maximum(0, Z)
        else:
            A_new = Z

    return A_new

W_new, B_new = train([128, 64, 32, 16], 0.0001, 15000)

parameters = {
    'W': W_new,
    'B': B_new,
}

with open(r'C:\Users\guilh\PycharmProjects\DashboardAi\Src\NeuralNetwork\ModelsAi\parameters_gen_6.pkl', 'wb') as p:
    pickle.dump(parameters, p)
    print("Parameters saved successfully")




