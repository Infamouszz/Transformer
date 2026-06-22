import io
import pickle
import chess.pgn
import numpy as np
import sys
from importlib import resources
import torch
import torch_directml

np.set_printoptions(threshold=sys.maxsize)

pgn = io.StringIO("1. e4 c5 2. Nf3 Nc6 3. Bc4 e6 4. d3 Na5 5. Nc3 Nxc4 6. dxc4 Be7 7. Bf4 a6 8. O-Ob6 9. Ne5 Nf6 10. Qf3 O-O 11. Rfd1 h6 12. Rd3 Ra7 13. Qh3 Nh7 14. Bxh6 gxh6 15.Rg3+ Kh8 16. Qxh6 Rg8 17. Nxf7# 1-0")
gamec = chess.pgn.read_game(pgn)
boardc = gamec.board()


pieces = {'r': 11, 'n': 10, 'b': 9,'k': 8, 'q': 7, 'p': 6, 'P': 5, 'Q': 4, 'K': 3, 'B': 2, 'N': 1, 'R': 0}
list_of_positions = {}

def board_to_dataset(board, index):
    matrix_board = np.zeros([1, 769])
    turn = board.turn

    for squares in range(0,63):
        piece = board.piece_at(squares)
        if piece is not None:
            piece_index = insert_piece(squares, piece in pieces)
            matrix_board[0, piece_index] = 1
    matrix_board[0, 768] = 1 if turn == chess.WHITE else -1
    list_of_positions[index] = matrix_board

def insert_piece(square, piece):
    piece_index = (((square + 1) * 12) - piece) - 1
    return piece_index

def run(game_node, current_index):
    board = game_node.board()

    for move in game_node.mainline_moves():
        board_to_dataset(board, current_index)
        board.push(move)
        current_index += 1
    return current_index

run(gamec, 0)

print("list_of_positions:", list_of_positions)

with resources.files('ModelsAi').joinpath('parameters_gen_10.pkl').open('rb') as f:
    loaded_parameters = pickle.load(f)

with resources.files('ModelsAi').joinpath('matrix_gen_10_Y.pkl').open('rb') as m:
    loaded_matrix = pickle.load(m)

Y = loaded_matrix

W_tensor = loaded_parameters['W']
B_tensor = loaded_parameters['B']

W = W_tensor
B = B_tensor

X_entered = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        1.]])
X_normalized = X_entered / np.max(X_entered)

def predict_game(X, W, B):
    A_new = X

    layers = len(W)

    for i in range(layers):
        Z = np.dot(A_new, W[i].cpu().numpy()) + B[i].cpu().numpy()

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
    return np.round(probabilities * 100)

result = predict_game(X_entered, W, B)
print("Predict:" ,result)

