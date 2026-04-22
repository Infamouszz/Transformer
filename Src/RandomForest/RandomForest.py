from numpy.ma.core import sqrt, count
import random as rnd
import BinaryTree as tree
import numpy as np
import pandas as pd
from collections import Counter
import math

X_treino = np.array([
    [25, 100, 0],
    [45, 70, 0],
    [30, 168, 1],
    [65, 130, 1],
    [22, 96, 0]
])
X_treino2 = pd.DataFrame(X_treino)
Y_treino = np.array(["Normal", "Normal", "Infartando", "Infartando", "Normal"]).reshape(-1, 1)
class RandomForest():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.trees = []

    def build_random_forest(self, max_tree = 100):
        for i in range(max_tree):
            boot = Bootstrap(self.X, self.Y)
            bootstraped_X, bootstraped_Y = boot.shuffle()
            tree_builder = tree.TreeBuilder()
            tree_builder.fit(bootstraped_X, bootstraped_Y)
            self.trees.append(tree_builder)
        return self.trees

    def predict_random_forest(self, X_real):
        votes = np.array([tree.predict(X_real) for tree in self.trees])

        final_proportion = [{str(classes): num_votes for classes, num_votes in Counter(sample_votes).items()} for sample_votes in votes.T]

        return final_proportion
class Bootstrap:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def shuffle(self):
        feature_samples = len(self.X)
        random_indexes = np.random.randint(0, feature_samples, size=feature_samples)
        bootstrapped_X = self.X[random_indexes]
        bootstrapped_Y = self.Y[random_indexes]
        return bootstrapped_X, bootstrapped_Y

forest = RandomForest(X_treino, Y_treino)
forest.build_random_forest()
new_value = np.array([[50, 170, 1]])
random_forest_result = forest.predict_random_forest(new_value)
print("Resultado random forest: ", random_forest_result)








