import pickle

import chess
import chess.pgn
import io
import numpy as np
import chess.engine as eng

engine = chess.engine.SimpleEngine.popen_uci(
    r"C:\Users\guilh\PycharmProjects\DashboardAi\stockfish\stockfish-windows-x86-64-avx2.exe")

games = {}
with open(r"C:\Users\guilh\PycharmProjects\DashboardAi\lichess_tournament_2026.04.16_0Guo5doZ_yearly-rapid.pgn", "r",
          encoding="utf-8") as pgn_file:
    counter = 0

    while counter < 1000:
        game = chess.pgn.read_game(pgn_file)

        if game is None:
            break

        games[counter] = game

        if counter > 0 and counter % 100 == 0 or counter == 1000:
            print("Games loaded: ", counter)

        counter += 1

pieces = {'p': -1, 'P': 1, 'b': -2, 'B': 2, 'n': -3, 'N': 3, 'r': -4, 'R': 4, 'q': -5, 'Q': 5, 'k': -6, 'K': 6}
list_of_positions = {}
list_of_evaluation = {}


def board_to_dataset(board, index):
    matrix_board = np.zeros([1, 65])
    matrix_evaluation = np.zeros([1,5])

    eval_info = engine.analyse(board, chess.engine.Limit(time=0.1))
    score = eval_info["score"].pov(board.turn)
    value = score.score(mate_score=1000) / 100.0

    if -1 <= value <= 1:
        matrix_evaluation[0,2] = 1
    elif 1 < value <= 3:
        matrix_evaluation[0,3] = 1
    elif value > 3:
        matrix_evaluation[0,4] = 1
    elif -3 <= value < -1:
        matrix_evaluation[0,1] = 1
    elif value < -3:
        matrix_evaluation[0,0] = 1

    list_of_evaluation[index] = matrix_evaluation


def run(game_node, current_index):
    board = game_node.board()

    for move in game_node.mainline_moves():
        board_to_dataset(board, current_index)
        board.push(move)
        current_index += 1
    return current_index


def transform_all_games():
    index = 0
    game_counter = 0
    for chess_game_key in games:
        index = run(games[chess_game_key], index)

        game_counter += 1
        print("Games processed: ", game_counter)

    return list_of_positions, list_of_evaluation

transform_all_games()

Y = {'Y':list_of_evaluation}

with open(r'C:\Users\guilh\PycharmProjects\DashboardAi\Src\NeuralNetwork\ModelsAi\matrix_y_gen_5.pkl', 'wb') as f:
    pickle.dump(Y, f)
    print("Parameters saved successfully")