from typing import Dict, Callable, Tuple

import torch

from . import networks
from .base_model import BaseModel
from .losses import binary_classification_accuracy, multiclass_accuracy
from ..options.classifier_options import ClassifierOptions


class ImageClassifier(BaseModel):

    def __init__(self, parser: ClassifierOptions):
        """Instanciate a ImageClassifier with the defined options.

        Args:
            options (ClassifierOptions)


        """
        super().__init__(parser)

        network_name = self.options.network
        loss_name = self.options.loss_function
        metric_name = self.options.metric

        if network_name == "resnet":
            model = networks.resnet18_classifier(self.n_labels)
        else:
            raise NotImplementedError(f"Network {network_name} has not been implmented")

        if loss_name == "BinaryCrossEntropy":
            loss_function = torch.nn.BCEWithLogitsLoss()
        elif loss_name == "CrossEntropy":
            loss_function = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Loss function {loss_name} has not been implmented")

        if metric_name == "Accuracy" and self.n_labels == 1:
            metric = binary_classification_accuracy(self.threshold)
        elif metric_name == "Accuracy" and self.n_labels > 1:
            metric = multiclass_accuracy
        else:
            raise NotImplementedError(f"Metric {metric_name} has not been implmented")

        self.models: Dict[str, torch.nn.Module] = {network_name: model.to(self.device)}
        self.criterions: Dict[str, Callable] = {network_name: loss_function}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {network_name: torch.optim.Adam(self.models[network_name].parameters(), lr=self.lr)}
        self.metric: Callable = metric

    @property
    def n_labels(self) -> str:
        return self.options.n_classes

    @property
    def lr(self) -> str:
        return self.options.learning_rate

    @property
    def threshold(self) -> float:
        return self.options.threshold

    def set_input(self, input: Dict[str, torch.Tensor]) -> None:
        """Set the batch of images and labels that will be used to update parameters or validating the model.

        Args:
            input (Dict[str, torch.Tensor]): keys=('images', 'labels')
        """

        self.image = input["images"].to(self.device)
        self.labels = input["labels"].to(self.device)

    def forward(self) -> None:
        """Applies a the forward step of the network
        """
        self.pred = self.models[self.options.network](self.image)

    def backward(self) -> float:
        """Applies the backward step of the model, computing the loss, the gradients and updating parameters.

        Returns:
            float: loss 
        """
        loss: torch.Tensor = self.criterions[self.options.network](self.pred, self.labels)
        self.optimizers[self.options.network].zero_grad()
        loss.backward()
        self.optimizers[self.options.network].step()

        return loss.cpu().item()

    def update_parameters(self) -> float:
        """Applyis the forward and backward steps and returns the loss

        Returns:
            float: loss 
        """

        super().update_parameters()  # Forward
        loss = self.backward()

        return loss

    def validation(self) -> Tuple[float, float]:
        """Applies a validation step over the inputs returning the loss and metric of the predictions.

        Returns:
            Tuple[float, float]: loss, metric
        """

        super().validation()

        with torch.no_grad():
            self.forward()
            loss: torch.Tensor = self.criterions[self.options.network](self.pred, self.labels)
            metric = self.metric(self.pred, self.labels)

        return loss.cpu().item(), metric

    def inference(self, input: torch.Tensor) -> None:

        super().inference(input)
        with torch.no_grad():
            predictions: torch.Tensor = self.models[self.options.network](input.to(self.device))

        if self.n_labels == 1:
            return predictions.cpu() > self.threshold
        else:
            return torch.argmax(predictions, dim=1).cpu()
