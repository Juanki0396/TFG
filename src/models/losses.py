

from typing import Callable
import torch


def binary_classification_accuracy(threshold: float = 0.5) -> Callable:
    """Returns accuracy function with the desired threshold

    Args:
        threshold (float, optional):  Defaults to 0.5.

    Returns:
        Callable: Accuracy function
    """
    def accuracy(predictions: torch.Tensor, truth: torch.Tensor) -> float:
        """Compute the correct classification by the predicitons. The threshold indicate the minimum probability
        needed to classify as positive the predictions.

        Args:
            predictions (torch.Tensor)
            truth (torch.Tensor)
            threshold (float, optional):  Defaults to 0.5.

        Returns:
            float: 
        """
        predictions = (predictions.reshape((-1)) > threshold).type(torch.float)
        truth = truth.reshape((-1))

        total = torch.eq(predictions, truth).mean().item()

        return total

    return accuracy


def multiclass_accuracy(predictions: torch.Tensor, truth: torch.Tensor) -> float:
    """Evaluates the classification accuracy  for multiclass problems. In this case no threshold 
    is need since the most likely class is chosen as the classfication result.

    Args:
        predictions (torch.Tensor): N_samplesxClasses tensor obtained from the model
        truth (torch.Tensor): N_samplesxClasses tensor with correct outputs

    Returns:
        float: percent of correct predictions
    """
    predictions = predictions.argmax(1).reshape(-1)
    truth = truth.reshape(-1)

    total = torch.eq(predictions, truth).mean().item()

    return total
