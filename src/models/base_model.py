
from abc import ABC, abstractmethod
import os
from typing import Any, Dict, List, Callable

import torch


class BaseModel(ABC):

    def __init__(self):
        """Instanciate the networks, criterions and optimizers.
        """
        self.name: str = None
        self.models: Dict[str, torch.nn.Module] = {}
        self.criterions: Dict[str, Callable] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.metric: Callable = None

    @abstractmethod
    def set_input(self, input: Dict[str, torch.Tensor]) -> None:
        """This method will store the inputs of the model needed for the 
        forward method

        Args:
            input (Dict[str, torch.Tensor])
        """
        pass

    @abstractmethod
    def forward(self) -> None:
        """Apply the forward step over the stored inputs and store the results.
        """
        pass

    @abstractmethod
    def update_parameters(self) -> Dict[str, float]:
        """Use the forward method and apply the backward step, updating the model weights.
        It also return in a dictionary the different losses obatined.

        Returns:
            Dict[str, float]: loss name with its value.
        """
        self.set_train_mode(True)
        self.forward()
        # Continue the implementation

    @abstractmethod
    def validation(self) -> Dict[str, Any]:
        """Apply the forward step and evaluate the losses and metrics over a validation example

        Returns:
            Dict[str, Any]: _description_
        """
        self.set_train_mode(False)
        self.forward()
        # Continue the implementation

    @abstractmethod
    def inference(self, input: Dict[str, torch.Tensor] = None) -> Dict[str, Any]:
        """Implement the inference of the model. Input is set to None since generative model don't
        need any input.

        Args:
            input (Dict[str, torch.Tensor], optional): Defaults to None.

        Returns:
            Dict[str, Any]: Model outputs
        """
        pass

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

    def save_model(self, save_dir: str) -> None:
        """Creates or overwrite a save model folder, where the state dict is saved,
        in the selected saviing direrctory.


        Args:
            save_dir (str): Directory where the model folder is created.
        """

        model_dir = os.path.join(save_dir, self.name)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        for key, model in self.models.items():
            path = os.path.join(model_dir, f"{key}.pth")
            torch.save(model.state_dict(), path)

    def load_model(self, model_dir: str) -> None:
        """Load model state dicts stored in the model directory

        Args:
            model_dir (str): Directory that contains the state dicts of the networks.
        """

        for file in os.scandir(model_dir):
            net_name = file.name.split(".")[0]
            self.models[net_name].load_state_dict(torch.load(file.path))
