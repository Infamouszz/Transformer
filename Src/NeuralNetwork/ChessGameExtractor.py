import os

os.environ["HF_TOKEN"] = "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"
import pickle
from pathlib import Path
import chess
import numpy as np
from datasets import load_dataset

models_dir = Path("ModelsAi")
models_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)
skiped_dataset = dataset.skip(30_000_000)
print("Samples loaded successfully")

PIECE_TO_IDX = {
    'k': 0, 'q': 1, 'r': 2, 'n': 3, 'b': 4, 'p': 5,
    'P': 6, 'B': 7, 'N': 8, 'R': 9, 'Q': 10, 'K': 11
}

list_of_positions = {}
list_of_evaluations = {}
list_of_indexes = []


def evaluation_to_dataset(evaluation, turn, index):
    matrix_evaluation = np.zeros([1, 5])
    value_evaluation = evaluation / 100

    if value_evaluation >= 3:
        matrix_evaluation[0, 0] = 1
        list_of_indexes.append(1)
    elif value_evaluation >= 1:
        matrix_evaluation[0, 1] = 1
        list_of_indexes.append(2)
    elif value_evaluation >= -1:
        matrix_evaluation[0, 2] = 1
        list_of_indexes.append(3)
    elif value_evaluation >= -3:
        matrix_evaluation[0, 3] = 1
        list_of_indexes.append(4)
    else:
        matrix_evaluation[0, 4] = 1
        list_of_indexes.append(5)

    list_of_evaluations[index] = matrix_evaluation


def board_to_dataset(board, turn, index):
    matrix_board = np.zeros([1, 769])

    for square in range(64):
        piece = board.piece_at(square)
        if piece is not None:
            idx = PIECE_TO_IDX[piece.symbol()]
            matrix_board[0, square * 12 + idx] = 1

    matrix_board[0, 768] = 1 if turn == chess.WHITE else -1
    list_of_positions[index] = matrix_board


target_samples = 100_000
index_samples = 0

for sample in skiped_dataset:
    board_sample = chess.Board(sample['fen'])
    turn_samples = board_sample.turn
    eval_samples = sample['cp']

    if eval_samples is not None:
        evaluation_to_dataset(eval_samples, turn_samples, index_samples)
    else:
        mate = sample['mate']
        if mate:
            evaluation_to_dataset(1000 if mate > 0 else -1000, turn_samples, index_samples)
        else:
            evaluation_to_dataset(0, turn_samples, index_samples)

    board_to_dataset(board_sample, turn_samples, index_samples)
    index_samples += 1

    if index_samples >= target_samples:
        break

classes, counts = np.unique(list_of_indexes, return_counts=True)
min_class_distribution = np.min(counts) if len(counts) > 0 else 0
print(f"Min Classes: {min_class_distribution}", f"Counts: {counts}")

with open(models_dir / 'matrix_X.pkl', 'wb') as fx:
    pickle.dump(list_of_positions, fx)

with open(models_dir / 'matrix_Y.pkl', 'wb') as fy:
    pickle.dump(list_of_evaluations, fy)

print("Finished saving ModelsAi/matrix_X.pkl and ModelsAi/matrix_Y.pkl")