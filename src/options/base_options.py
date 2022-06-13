import abc
import argparse
import os
from typing import Any, Union


class BaseOptions(abc.ABC):

    def __init__(self):
        """Instanciate a BaseOptions object by reading and parsing programm parameters
        """
        self.parser = argparse.ArgumentParser()

    def isnotebook(self) -> bool:
        try:
            shell = get_ipython().__class__.__name__  # Seems to be in the global name space when ipython is running
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return True  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False      # Probably standard Python interpreter

    @abc.abstractmethod
    def read_parameters(self) -> None:
        """Abstract method that defines model options.
        """
        # General options
        self.parser.add_argument("--name", type=str, default="model", help="Set model name")
        self.parser.add_argument("--device", type=str, default="cpu", help="Set the device that pytorch will use for the model: (cpu, cuda:0, ...)")
        self.parser.add_argument("--saved_models_dir", type=str, default="./saved_models")

    def gather_options(self, force: bool = False) -> None:
        """Parse args to store them in the object
        """
        if self.isnotebook() or force:
            self.options = self.parser.parse_args("")
        else:
            self.options = self.parser.parse_args()

    def rewrite_option(self, option_name: str, value: Any) -> None:
        if hasattr(self.options, option_name):
            setattr(self.options, option_name, value)
        else:
            raise ValueError(f"Option {option_name} does not exist")

    def save_options(self, dir_path: str) -> None:

        opt = self.options

        save_path = os.path.join(dir_path, "options.txt")

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(save_path, "wt") as save_file:
            save_file.write(self.print_options(return_str=True))

    def print_options(self, return_str: bool = False) -> Union[None, str]:

        text = "MODEL OPTIONS".center(60, "-") + "\n"
        for atribute, value in self.options.__dict__.items():
            text += f"{atribute.capitalize():<25}---->{str(value):>25}\n"

        if return_str:
            return text

        print(text)
