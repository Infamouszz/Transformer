import torch

class ResidualConnectionCalculator:
    def __init__(self):
        pass

    @staticmethod
    def calculate(entered_X, multi_head_output):
        return entered_X + multi_head_output