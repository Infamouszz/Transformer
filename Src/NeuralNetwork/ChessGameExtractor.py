import pickle
import random
import os
os.environ["HF_TOKEN"] = "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"
import chess
import numpy as np
from datasets import load_dataset
from importlib import resources
from tqdm import tqdm

dataset = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)
print("Samples loaded successfully")

pieces = {'r': 11, 'n': 10, 'b': 9,'k': 8, 'q': 7, 'p': 6, 'P': 5, 'Q': 4, 'K': 3, 'B': 2, 'N': 1, 'R': 0}
#[r, n, b, k, q, p, P, Q, K, B, N, R]#
#[0,1,2,3,4,5,6,7,8,9,10,11][12,13,14,15,16,17,18,19,20,21,22,23]
list_of_positions = {}
list_of_evaluations = {}
list_of_indexes = []

def evaluation_to_dataset(evaluation, turn, index):
    matrix_evaluation = np.zeros([1,5])
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

    for squares in range(64):
        piece = board.piece_at(squares)
        if piece is not None:
            piece_symbol = piece.symbol()
            piece_index = insert_piece(squares, pieces[piece_symbol])
            matrix_board[0, piece_index] = 1
    matrix_board[0, 768] = 1 if turn == chess.WHITE else -1
    list_of_positions[index] = matrix_board

def insert_piece(square, piece):
    piece_index = (((square + 1) * 12) - piece) - 1
    return piece_index

def pruning(list_of_indexes_p, min_class_quantity):
    indexes_array = np.array(list_of_indexes_p)
    classes_to_sample = [1,2,3,4,5]
    pruned_indexes = []

    for c in classes_to_sample:
        class_indexes = np.where(indexes_array == c)[0]
        sampled = np.random.choice(class_indexes, size=min_class_quantity, replace=False)
        pruned_indexes.append(sampled)

    pruned_indexes = np.concatenate(pruned_indexes)
    np.random.shuffle(pruned_indexes)
    return set(pruned_indexes)


index_samples = 0
batches = 250_000
max_batches = 20
batch_counts = 0

for sample in tqdm(dataset, desc="Transforming samples..."):
    board_sample = chess.Board(sample['fen'])
    turn_samples = board_sample.turn
    eval_samples = sample.get('cp')
    mate = sample.get('mate')
    if mate is not None:
        eval_value = 10000 if mate > 0 else -10000
        evaluation_to_dataset(eval_value, turn_samples, index_samples)
    elif eval_samples is not None:
        evaluation_to_dataset(eval_samples, turn_samples, index_samples)
    else:
        print("Error")
        continue
    board_to_dataset(board_sample, turn_samples, index_samples)

    if index_samples == batches:
        with resources.files('TrainDatasets').joinpath(f'matrix_X_batch_{batch_counts}.pkl').open('wb') as p:
            pickle.dump(list_of_positions, p)

        with resources.files('TrainDatasets').joinpath(f'matrix_Y_batch_{batch_counts}.pkl').open('wb') as e:
            pickle.dump(list_of_evaluations, e)

        list_of_positions = {}
        list_of_evaluations = {}

        index_samples = 0
        print("Actual batch: ",batch_counts)
        batch_counts += 1
    else:
        index_samples += 1



    if batch_counts == max_batches:
        break

classes, counts = np.unique(list_of_indexes, return_counts=True)
min_class_distribution = np.min(np.unique(counts))
print(f"Min Classes {min_class_distribution}")

pruned_shuffled_indexes = pruning(list_of_indexes, min_class_distribution)

full_X = {}
full_Y = {}
global_index = 0
final_dict_index = 0
for j in tqdm(range(max_batches), desc="Pruning..."):
    with resources.files('TrainDatasets').joinpath(f'matrix_X_batch_{j}.pkl').open('rb') as p:
        temp_loaded_X = pickle.load(p)
    with resources.files('TrainDatasets').joinpath(f'matrix_Y_batch_{j}.pkl').open('rb') as e:
        temp_loaded_Y = pickle.load(e)

    for k in range(len(temp_loaded_X)):
        if global_index in pruned_shuffled_indexes:
            full_X[final_dict_index] = temp_loaded_X[k]
            full_Y[final_dict_index] = temp_loaded_Y[k]
            final_dict_index += 1
        global_index += 1

with resources.files('ModelsAi').joinpath(f'matrix_gen_10_X.pkl').open('wb') as fx:
    pickle.dump(full_X, fx)

with resources.files('ModelsAi').joinpath(f'matrix_gen_10_Y.pkl').open('wb') as fy:
    pickle.dump(full_Y, fy)

print("Matrix saved successfully!")