
from abc import ABC, abstractmethod
import os
from typing import Any, Dict, List, Callable

import torch


class BaseModel(ABC):

    def __init__(self):

        self.name: str = None
        self.models: Dict[str, torch.nn.Module] = {}
        self.criterions: Dict[str, Callable] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.metric: Callable = None

    @abstractmethod
    def set_input(self, input: Dict[str, torch.Tensor]) -> None:
        pass

    @abstractmethod
    def load_model(self, model_dir: str) -> None:
        pass

    @abstractmethod
    def save_model(self, save_dir: str) -> None:
        pass

    @abstractmethod
    def forward(self) -> None:
        pass

    @abstractmethod
    def update_parameters(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def test(self) -> Dict[str, Any]:
        pass

    def set_requires_grad(self, nets: List[torch.nn.Module], requires_grad=True) -> None:
        for net in nets:
            for par in net.parameters():
                par.requires_grad = requires_grad

    def set_train_mode(self, train_mode: bool) -> None:

        if train_mode:
            for model in self.models.values():
                model.train()
        else:
            for model in self.models.values():
                model.eval()

    def save_model(self, save_dir: str) -> None:

        model_dir = os.path.join(save_dir, self.name)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        for key, model in self.models.items():
            path = os.path.join(model_dir, f"{key}.pth")
            torch.save(model.state_dict(), path)

    def load_model(self, model_dir: str) -> None:

        for file in os.scandir(model_dir):
            net_name = file.name.split(".")[0]
            self.models[net_name].load_state_dict(torch.load(file.path))
