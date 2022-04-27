

import torch


def accuracy(threshold: float = 0.5):

    def metric(predictions: torch.Tensor, truth: torch.Tensor) -> float:
        predictions = predictions.reshape((-1))
        truth = truth.reshape((-1))

        n = len(predictions)
        total = ((predictions > threshold) == truth).type(torch.float).sum().item() / n

        return total
    return metric
