import pickle
import random
import os
os.environ["HF_TOKEN"] = "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"
import chess
import numpy as np
from datasets import load_dataset
from importlib import resources


dataset = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)
print("Samples loaded successfully")

pieces = {'p': -1, 'P': 1, 'b': -2, 'B': 2, 'n': -3, 'N': 3, 'r': -4, 'R': 4, 'q': -5, 'Q': 5, 'k': -6, 'K': 6}
list_of_positions = {}
list_of_evaluations = {}
list_of_indexes = []

def evaluation_to_dataset(evaluation, turn, index):
    matrix_evaluation = np.zeros([1,5])
    value_evaluation = evaluation / 100

    if value_evaluation >= 3:
        matrix_evaluation[0, 0] = 1
        list_of_indexes.insert(index,1)
    elif value_evaluation >= 1:
        matrix_evaluation[0, 1] = 1
        list_of_indexes.insert(index,2)
    elif value_evaluation >= -1:
        matrix_evaluation[0, 2] = 1
        list_of_indexes.insert(index,3)
    elif value_evaluation >= -3:
        matrix_evaluation[0, 3] = 1
        list_of_indexes.insert(index,4)
    else:
        matrix_evaluation[0, 4] = 1
        list_of_indexes.insert(index,5)

    list_of_evaluations[index] = matrix_evaluation

def board_to_dataset(board, turn, index):
    matrix_board = np.zeros([1, 65])

    for squares in range(0,63):
        piece = board.piece_at(squares)
        if piece is not None:
            matrix_board[0,squares] = pieces[piece.symbol()]
    matrix_board[0, 64] = 1 if turn == chess.WHITE else -1
    list_of_positions[index] = matrix_board

def pruning(list_of_indexes_p, min_class_quantity):
    class_A = np.random.choice(np.where(np.array(list_of_indexes) == 1)[0], size=min_class_quantity, replace=False)
    class_B = np.random.choice(np.where(np.array(list_of_indexes) == 2)[0], size=min_class_quantity, replace=False)
    class_C = np.random.choice(np.where(np.array(list_of_indexes) == 3)[0], size=min_class_quantity, replace=False)
    class_D = np.random.choice(np.where(np.array(list_of_indexes) == 4)[0], size=min_class_quantity, replace=False)
    class_E = np.random.choice(np.where(np.array(list_of_indexes) == 5)[0], size=min_class_quantity, replace=False)

    list_of_pruned_indexes = np.concatenate([class_A, class_B, class_C, class_D, class_E])
    np.random.shuffle(list_of_pruned_indexes)
    return list_of_pruned_indexes


index_samples = 0
batches = 10_000
max_batches = 20
batch_counts = 0

for sample in dataset:
    board_sample = chess.Board(sample['fen'])
    turn_samples = board_sample.turn
    eval_samples = sample['cp']
    if eval_samples and board_sample is not None:
        evaluation_to_dataset(eval_samples, turn_samples, index_samples)
    else:
        mate = sample['mate']
        if mate:
            evaluation_to_dataset(1000, turn_samples, index_samples)
        elif eval_samples == 0:
            evaluation_to_dataset(0, turn_samples, index_samples)
        else:
            print("Error")
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
print("Indexes: ",list_of_indexes)
print(f"Min Classes {min_class_distribution}",f"Counts {counts}")

pruned_shuffled_indexes = pruning(list_of_indexes, min_class_distribution)

full_X = {}
full_Y = {}

for j in range(batches):
    with resources.files('TrainDatasets').joinpath(f'matrix_X_batch_{max_batches-1}.pkl').open('rb') as p:
        temp_loaded_X = pickle.load(p)
    with resources.files('TrainDatasets').joinpath(f'matrix_Y_batch_{max_batches-1}.pkl').open('rb') as e:
        temp_loaded_Y = pickle.load(e)
    if j in pruned_shuffled_indexes:
        full_X[j] = temp_loaded_X[j]
        full_Y[j] = temp_loaded_Y[j]
    j += 1
    if j == batches:
        j = 0
        temp_loaded_X = {}
        temp_loaded_Y = {}
        max_batches -= 1

    if max_batches == 0:
        break

print("Finished")
print("X: ", full_X)
print("Y: ", full_Y)

with resources.files('ModelsAi').joinpath(f'matrix_gen_8_X.pkl').open('wb') as fx:
    pickle.dump(full_X, fx)

with resources.files('ModelsAi').joinpath(f'matrix_gen_8_Y.pkl').open('wb') as fy:
    pickle.dump(full_Y, fy)