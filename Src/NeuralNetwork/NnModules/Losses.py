import torch

class CrossEntropyLoss:
    def __init__(self, eps=1e-15):
        self.eps = eps

    def foward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.eps, 1 - self.eps)
        loss = -torch.sum(y_true * torch.log(y_pred)) / y_pred.shape[0]
        return loss

    def derivative(self, y_pred, y_true):
        batch_m = y_pred.shape[0]
        dZo = (y_pred - y_true)/batch_m
        return dZo