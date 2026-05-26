import io
import pickle
import chess.pgn
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

pgn = io.StringIO("1.e4 e5 2. d3 Nc6 3. Nc3 Nf6 4. Be3 Bb4 5. a3 Bxc3+ 6. bxc3 O-O 7. Qd2 h6 8.O-O-O $2 d5 9. Nf3 d4 10. cxd4 exd4 11. Nxd4 $6 Nxd4 12. Bf4 Be6 13. Qb4 Qd7 14.c3 $6 Nb3+ 15. Kc2 a5 16. Qxb7 Rab8 17. Qa7 Nxe4 $3 18. dxe4 Na1+ $3 19. Kc1 $4Rb1+ $3 20. Kxb1 Qxd1+ $1 21. Bc1 Qc2+ 22. Kxa1 Qxc1# 0-1")
gamec = chess.pgn.read_game(pgn)
board = gamec.board()


with open(r'C:\Users\guilh\PycharmProjects\DashboardAi\Src\NeuralNetwork\ModelsAi\parameters_gen_6.pkl', 'rb') as f:
    loaded_parameters = pickle.load(f)

with open(r'C:\Users\guilh\PycharmProjects\DashboardAi\Src\NeuralNetwork\ModelsAi\matrix_y_gen_5.pkl', 'rb') as m:
    loaded_matrix = pickle.load(m)

Y = loaded_matrix['Y']

W = loaded_parameters['W']
B = loaded_parameters['B']

X_entered = np.array([[-3.,  6.,  2.,  0.,  0.,  2.,  0.,  4.,  0.,  0., -5.,  0.,  0.,
         1.,  1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0., -2.,  0.,  0., -1.,  5.,  0., -1.,  0.,
         0., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0., -4., -6.,  0.,  1.]])
X_normalized = X_entered / np.max(X_entered)

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

