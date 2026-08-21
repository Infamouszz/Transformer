import pickle
import random
import os
from importlib import resources

import chess
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


dataset = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)
print("Samples loaded successfully")

skiped_dataset = dataset.skip(30_000_000).take(100_000)
list_of_evaluations = []

def evaluation_to_dataset(evaluation):
    matrix_evaluation = np.zeros([1, 5])
    value_evaluation = evaluation / 100

    if value_evaluation >= 3:
        matrix_evaluation[0, 0] = 1
    elif value_evaluation >= 1:
        matrix_evaluation[0, 1] = 1
    elif value_evaluation >= -1:
        matrix_evaluation[0, 2] = 1
    elif value_evaluation >= -3:
        matrix_evaluation[0, 3] = 1
    else:
        matrix_evaluation[0, 4] = 1

    return matrix_evaluation


for sample in tqdm(skiped_dataset, desc="Transforming dataset"):
    board_sample = chess.Board(sample['fen'])
    turn_samples = board_sample.turn
    eval_samples = sample['cp']

    if eval_samples is not None:
        mat = evaluation_to_dataset(eval_samples)
    else:
        mate = sample['mate']
        if mate is not None:
            val = 1000 if mate > 0 else -1000
            mat = evaluation_to_dataset(val)
        else:
            mat = evaluation_to_dataset(0)

    list_of_evaluations.append(mat)

print(f"Processados {len(list_of_evaluations)} samples com sucesso!")

with resources.files('ModelsAi').joinpath(f'test_matrix.pkl').open('wb') as fx:
    pickle.dump(list_of_evaluations, fx)