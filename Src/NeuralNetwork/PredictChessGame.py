import io
import pickle
import chess.pgn
import numpy as np
import sys
from importlib import resources

np.set_printoptions(threshold=sys.maxsize)

with resources.files('ModelsAi').joinpath('parameters_gen_10.pkl').open('rb') as f:
    loaded_parameters = pickle.load(f)

with open('matrix_X.pkl','rb') as x:
    X = pickle.load(x)

with open('matrix_Y.pkl',   'rb') as y:
    Y = pickle.load(y)

W = [w.cpu().numpy() for w in loaded_parameters['W']]
B = [b.cpu().numpy() for b in loaded_parameters['B']]

def predict_game(X, W, B):
    A_new = X
    layers = len(W)

    for i in range(layers):
        Z = np.dot(A_new, W[i]) + B[i]

        if i < layers - 1:
            A_new = np.maximum(0, Z)
        else:
            A_new = Z

    return A_new

hits = 0
total = len(X)

for key in X.keys():
    x_sample = X[key]
    y_true = Y[key]

    prediction = predict_game(x_sample, W, B)

    pred_class = np.argmax(prediction)
    true_class = np.argmax(y_true)

    if pred_class == true_class:
        hits += 1

accuracy = (hits / total) * 100
print(f"Hits: {hits}/{total} ({accuracy:.2f}%)")
