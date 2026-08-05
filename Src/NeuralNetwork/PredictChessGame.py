import io
import pickle
import chess.pgn
import numpy as np
import sys
from importlib import resources

np.set_printoptions(threshold=sys.maxsize)

pgn = io.StringIO(
    "1. e4 e5 2. Nf3 Nc6 3. Bc4 h6 4. d3 Nf6 5. O-O d6 6. d4 exd4 7. Nxd4 Bd7 8. Nc3 a6 9. Nf5 Bxf5 10. exf5 Be7 11. Re1 O-O 12. Ne4 Nxe4 13. Rxe4 b5 $2 14. Bd5 $1 Qd7 $2 15. Bxc6 Qxc6 16. Rxe7 Rfe8 17. f6 g5 18. Qh5 1/2-1/2")
gamec = chess.pgn.read_game(pgn)

pieces = {'r': 11, 'n': 10, 'b': 9, 'k': 8, 'q': 7, 'p': 6, 'P': 5, 'Q': 4, 'K': 3, 'B': 2, 'N': 1, 'R': 0}
list_of_positions = {}


def insert_piece(square, piece_value):
    piece_index = (((square + 1) * 12) - piece_value) - 1
    return piece_index


def board_to_dataset(board, index):
    matrix_board = np.zeros([1, 769])
    turn = board.turn

    for square in range(64):
        piece = board.piece_at(square)
        if piece is not None:
            piece_symbol = piece.symbol()
            if piece_symbol in pieces:
                piece_value = pieces[piece_symbol]
                piece_index = insert_piece(square, piece_value)
                matrix_board[0, piece_index] = 1

    matrix_board[0, 768] = 1 if turn == chess.WHITE else -1
    list_of_positions[index] = matrix_board


def run(game_node, current_index):
    board = game_node.board()

    for move in game_node.mainline_moves():
        board_to_dataset(board, current_index)
        board.push(move)
        current_index += 1
    return current_index


run(gamec, 0)

with resources.files('ModelsAi').joinpath('parameters_gen_10.pkl').open('rb') as f:
    loaded_parameters = pickle.load(f)

W = [w.cpu().numpy() for w in loaded_parameters['W']]
B = [b.cpu().numpy() for b in loaded_parameters['B']]


def softmax(Z):
    Z_clamp = np.clip(Z, a_min=-80.0, a_max=80.0)
    Z_shifted = Z_clamp - np.max(Z_clamp, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    probabilities = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    return np.round(probabilities)


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
key = 0
for key, X_matrix in list_of_positions.items():
    max_val = np.max(X_matrix)
    X_normalized = X_matrix / (max_val if max_val > 0 else 1)
    temp_result = predict_game(X_normalized, W, B)

    print(f"Position: {key} | Predict: {temp_result}")