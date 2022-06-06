import abc
import argparse
import os
from typing import Any


class BaseOptions(abc.ABC):

    def __init__(self):
        """Instanciate a BaseOptions object by reading and parsing programm parameters
        """
        self.parser = argparse.ArgumentParser()

    @abc.abstractmethod
    def read_parameters(self) -> None:
        """Abstract method that defines model options.
        """
        # General options
        self.parser.add_argument("--name", type=str, default="model", help="Set model name")
        self.parser.add_argument("--device", type=str, default="cpu", help="Set the device that pytorch will use for the model: (cpu, cuda:0, ...)")
        self.parser.add_argument("--saved_models_dir", type=str, default="./saved_models")

    def gather_options(self) -> None:
        """Parse args to store them in the object
        """
        self.options = self.parser.parse_args()

    def rewrite_option(self, option_name: str, value: Any) -> None:
        if hasattr(self.options, option_name):
            setattr(self.options, option_name, value)
        else:
            raise ValueError(f"Option {option_name} does not exist")

    def save_options(self) -> None:

        opt = self.options
        saving_dir = os.path.join(opt.saved_models_dir, opt.name)
        save_path = os.path.join(saving_dir, "options.txt")

        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)

        with open(save_path, "wt") as save_file:
            text = "MODEL OPTIONS".center(100, "-") + "\n"
            for key, option in vars(opt).items():
                text += f"{key}: {option}\n"
            save_file.write(text)
