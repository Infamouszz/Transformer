import math
import pandas as pd
import numpy as np
import random as rnd

from numpy.ma.core import sqrt
class Node():
    def __init__(self, threshold: object = None, winner_feature_index: object = None, left_pointer: object = None, right_pointer: object = None,
                 info_gain: object = None,
                 value: object = None) -> None:
        self.threshold = threshold
        self.winner_feature_index = winner_feature_index
        self.left_pointer = left_pointer
        self.right_pointer = right_pointer
        self.info_gain = info_gain
        self.value = value
class TreeBuilder():
    def __init__(self, min_samples_split = 2, max_depth = 2):
        self.root = None

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, current_depth = 0):
        X,Y = dataset[:,:-1],dataset[:,-1]
        num_samples, num_features = np.shape(X)
        if num_samples >= self.min_samples_split and current_depth <= self.max_depth:
            best_split_getter = Best_Split()
            best_split = best_split_getter.get_best_split(dataset, num_features)
            if best_split and best_split.get("info_gain", 0) > 0:
                left_subtree = self.build_tree(best_split["dataset_left"], current_depth + 1)
                right_subtree = self.build_tree(best_split["dataset_right"], current_depth + 1)
                return Node(
                    threshold=best_split["threshold"],
                    winner_feature_index=best_split["feature_index"],
                    left_pointer=left_subtree,
                    right_pointer=right_subtree,
                    info_gain=best_split["info_gain"]
                )
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)

    def calculate_leaf_value(self, Y):
        Y = list(Y)
        winner_class = max(set(Y), key=Y.count)
        return winner_class

    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X_real):
        predictions = [self.make_prediction(x, self.root) for x in X_real]
        return predictions

    def make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value

        indice_inteiro = int(tree.winner_feature_index)

        feature_val = x[indice_inteiro]

        if float(feature_val) <= float(tree.threshold):
            return self.make_prediction(x, tree.left_pointer)
        else:
            return self.make_prediction(x, tree.right_pointer)

class Information_Gain():
    def get_information_gain(self, parent, l_child, r_child, mode="entropy"):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == "gini":
            gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain

    def entropy(self, y):
        classes, counters = np.unique(y, return_counts=True)
        proportion = counters/len(y)
        entropy = -np.sum(proportion * np.log2(proportion))
        return entropy
    def gini_index(self, y):
        classes, counters = np.unique(y, return_counts=True)
        proportion = counters/len(y)
        gini = 1.0 - np.sum(proportion**2)
        return gini
class Best_Split():
    def get_best_split(self, dataset, num_features):
        best_split = {}
        max_info_gain = -float('inf')
        picked_features = self.draw_features(num_features)

        for feature_index in picked_features:
            feature_value = dataset[:,feature_index]
            possible_thresholds = self.get_possible_thresholds(feature_value)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    parent_y, child_y_left, child_y_right = dataset[:,-1], dataset_left[:,-1], dataset_right[:,-1]
                    info_gain = Information_Gain()
                    curr_info_gain = info_gain.get_information_gain(parent_y, child_y_left, child_y_right, "gini")
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split

    def draw_features(self, num_features):

        features_limit = max(1, math.floor(math.sqrt(num_features)))
        picked_features = rnd.sample(range(0, num_features), features_limit)

        return picked_features

    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if float(row[feature_index]) <= threshold])
        dataset_right = np.array([row for row in dataset if float(row[feature_index]) > threshold])
        return dataset_left, dataset_right

    def get_possible_thresholds(self, feature_values):
        feature_values = feature_values.astype(float)
        unique_values = np.unique(feature_values)
        thresholds_array = (unique_values[:-1] + unique_values[1:]) / 2.0
        return thresholds_array.tolist()








