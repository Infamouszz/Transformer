import io
import pickle
import chess.pgn
import numpy as np
import sys
from importlib import resources
import torch

np.set_printoptions(threshold=sys.maxsize)

pgn = io.StringIO("1. e4 c5 2. Nf3 Nc6 3. Bc4 e6 4. d3 Na5 5. Nc3 Nxc4 6. dxc4 Be7 7. Bf4 a6 8. O-Ob6 9. Ne5 Nf6 10. Qf3 O-O 11. Rfd1 h6 12. Rd3 Ra7 13. Qh3 Nh7 14. Bxh6 gxh6 15.Rg3+ Kh8 16. Qxh6 Rg8 17. Nxf7# 1-0")
gamec = chess.pgn.read_game(pgn)
board = gamec.board()


with resources.files('ModelsAi').joinpath('parameters_gen_6.pkl').open('rb') as f:
    loaded_parameters = pickle.load(f)

with resources.files('ModelsAi').joinpath('matrix_gen_5.pkl').open('rb') as m:
    loaded_matrix = pickle.load(m)

Y = loaded_matrix['Y']

W = loaded_parameters['W']
B = loaded_parameters['B']

X_entered = np.array([[ 4.,  0.,  0.,  0.,  0.,  0.,  6.,  0.,  1.,  1.,  1.,  0.,  0.,
         1.,  1.,  1.,  0.,  0.,  3.,  0.,  0.,  0.,  4.,  0.,  0.,  0.,
         1.,  0.,  1.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  3.,  0.,  0.,
         0., -1., -1.,  0.,  0., -1.,  0.,  0.,  5., -4.,  0.,  0., -1.,
        -2., -1.,  0., -3.,  0.,  0., -2., -5.,  0., -4.,  0., -6.,  0.]])
X_normalized = X_entered / np.max(X_entered)

def predict_game(X, W, B):
    A_new = X

    layers = len(W)

    for i in range(layers):
        Z = np.dot(A_new, W[i]) + B[i]

        if i < layers - 1:
            A_new = np.maximum(0, Z)
        else:
            A_new = softmax(Z)

    return A_new

def softmax(Z):
    Z_clamp = np.clip(Z, min = -80.0, max = 80.0)

    Z_shifted = Z_clamp - np.max(Z_clamp, axis = 1, keepdims=True)

    exp_Z = np.exp(Z_shifted)
    probabilities = exp_Z / np.sum(exp_Z, axis = 1, keepdims = True)
    return probabilities

result = predict_game(X_entered, W, B)
print("Predict:" ,result)


pieces = {'p': -1, 'P': 1, 'b': -2, 'B': 2, 'n': -3, 'N': 3, 'r': -4, 'R': 4, 'q': -5, 'Q': 5, 'k': -6, 'K': 6}
list_of_positions = {}


def board_to_dataset(board, index):
    matrix_board = np.zeros([1, 65])
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            matrix_board[0, square] = pieces[piece.symbol()]

    matrix_board[0, 64] = 1 if board.turn else 0
    list_of_positions[index] = matrix_board

def run(game_node, current_index):
    board = game_node.board()

    for move in game_node.mainline_moves():
        board_to_dataset(board, current_index)
        board.push(move)
        current_index += 1
    return current_index

run(gamec, 0)

print("list_of_positions:", list_of_positions)

