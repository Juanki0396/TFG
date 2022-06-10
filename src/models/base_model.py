
from abc import ABC, abstractmethod
import os
from typing import Any, Dict, List, Callable, Union

import torch

from ..options.base_options import BaseOptions


class BaseModel(ABC):

    def __init__(self, parser: BaseOptions):
        """Instanciate the networks, criterions and optimizers.
        """

        self.parser = parser
        self.options = parser.options

        self.models: Dict[str, torch.nn.Module] = {}
        self.criterions: Dict[str, Callable] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.metric: Callable = None

    @property
    def name(self) -> str:
        return self.options.name

    @property
    def save_path(self) -> str:
        return os.path.join(self.options.saved_models_path, self.name)

    @property
    def device(self) -> str:
        return self.options.device

    @abstractmethod
    def set_input(self, input: Dict[str, torch.Tensor]) -> None:
        """Abstarct method that will store the inputs of the model needed for the 
        forward method

        Args:
            input (Dict[str, torch.Tensor])
        """
        pass

    @abstractmethod
    def forward(self) -> None:
        """Abstract method that applies the forward step over the stored inputs and store the results.
        """
        pass

    @abstractmethod
    def update_parameters(self) -> None:
        """Abstract method that use the forward method and apply the backward step, updating the model weights.
        It also return in a dictionary the different losses obatined.
        """
        self.set_train_mode(True)
        self.forward()
        # Continue the implementation

    @abstractmethod
    def validation(self) -> None:
        """Apply the forward step and evaluate the losses and metrics over a validation example 
        """
        self.set_train_mode(False)
        # Continue the implementation

    @abstractmethod
    def inference(self, input: Union[Dict[str, torch.Tensor], torch.Tensor]) -> None:
        """Implement the inference of the model. 
        Args:
            input (Dict[str, torch.Tensor] | torch.Tensor)
        """
        self.set_train_mode(False)
        # Continue the implementation

    def set_requires_grad(self, nets: List[torch.nn.Module], requires_grad=True) -> None:
        """Activate or deactivate the gradient computation of the selected nets.

        Args:
            nets (List[torch.nn.Module]): 
            requires_grad (bool, optional): Defaults to True.
        """
        for net in nets:
            for par in net.parameters():
                par.requires_grad = requires_grad

    def set_train_mode(self, train_mode: bool) -> None:
        """Change the mode of the model between train or eval.

        Args:
            train_mode (bool): True -> train; False -> eval
        """
        if train_mode:
            for model in self.models.values():
                model.train()
        else:
            for model in self.models.values():
                model.eval()

    def save_model(self, save_dir: str = None) -> None:
        """Creates or overwrite a save model folder, where the state dict is saved,
        in the selected saviing direrctory.


        Args:
            save_dir (str): Directory where the model folder is created.
        """

        if save_dir is None:
            save_dir = os.path.join(self.options.saved_models_dir, self.options.name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for key, model in self.models.items():
            path = os.path.join(save_dir, f"{key}.pth")
            torch.save(model.state_dict(), path)
            self.parser.save_options(save_dir)

    def load_model(self, model_dir: str) -> None:
        """Load model state dicts stored in the model directory

        Args:
            model_dir (str): Directory that contains the state dicts of the networks.
        """

        for file in os.scandir(model_dir):
            net_name = file.name.split(".")[0]
            self.models[net_name].load_state_dict(torch.load(file.path))
