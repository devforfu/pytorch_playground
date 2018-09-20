import torch


def accuracy(y_pred, y_true):
    match = y_pred.argmax(dim=1) == y_true
    acc = match.type(torch.float).mean()
    return acc.item()
