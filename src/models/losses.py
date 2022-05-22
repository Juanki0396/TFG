

import torch


def binary_classification_accuracy(predictions: torch.Tensor, truth: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute the correct classification by the predicitons. The threshold indicate the minimum probability
    needed to classify as positive the predictions.

    Args:
        predictions (torch.Tensor)
        truth (torch.Tensor)
        threshold (float, optional):  Defaults to 0.5.

    Returns:
        float: 
    """
    predictions = predictions.reshape((-1))
    truth = truth.reshape((-1))

    n = len(predictions)
    total = ((predictions > threshold) == truth).type(torch.float).sum().item() / n

    return total
