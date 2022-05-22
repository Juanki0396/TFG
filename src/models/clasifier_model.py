from typing import Dict, Callable

import torch

from . import networks
from .base_model import BaseModel
from .losses import accuracy


class ImageClassifier(BaseModel):

    def __init__(self, lr: float, threshold: float, device: str, name: str):
        super().__init__()

        self.name = name
        self.models: Dict[str, torch.nn.Module] = {"resnet": networks.resnet18_classifier(1).to(device)}
        self.criterions: Dict[str, Callable] = {"resnet": torch.nn.BCELoss()}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {
            "resnet": torch.optim.Adam(self.models["resnet"].parameters(), lr=lr)}
        self.metric: Callable = accuracy(threshold)
        self.device = device
        self.sigmoid = torch.nn.Sigmoid()

    def set_input(self, input: torch.Tensor, labels: torch.Tensor) -> None:
        self.image = input.to(self.device)
        self.labels = labels.to(self.device)

    def forward(self) -> None:

        self.pred = self.models["resnet"](self.image)
        self.pred = self.sigmoid(self.pred)

    def backward(self):

        loss = self.criterions["resnet"](self.pred, self.labels)
        self.optimizers["resnet"].zero_grad()
        loss.backward()
        self.optimizers["resnet"].step()

        return loss.item()

    def update_parameters(self) -> Dict[str, float]:

        self.set_train_mode(True)
        self.forward()
        loss = self.backward()

        return {"BCEloss": loss}

    def test(self) -> Dict[str, float]:

        self.set_train_mode(False)
        self.forward()
        loss = self.criterions["resnet"](self.pred, self.labels)
        metric = self.metric(self.pred, self.labels)

        return {"BCEloss": loss.item(), "Accuracy": metric}
