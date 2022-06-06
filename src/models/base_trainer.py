
import abc

from torch.utils.data import Dataset, DataLoader

from .base_model import BaseModel


class BaseTrainer(abc.ABC):

    def __init__(self, model: BaseModel) -> None:
        """Instanciate a trainer by selecting the model that will be trained

        Args:
            model (BaseModel): model to train
        """

        super().__init__()
        self.model = model

    @abc.abstractmethod
    def set_dataloaders(self, training_dataset: Dataset, validation_dataset: Dataset) -> None:
        """Set the Dataloaders from the datasets passed

        Args:
            training_dataset (Dataset): data to use in training
            validation_dataset (Dataset): data to use in validating
        """
        try:
            batch_size = self.model.options.batch_size
            num_workers = self.model.options.num_threads
        except NameError:
            batch_size = 1
            num_workers = 0

        self.traning_dataloader = DataLoader(training_dataset, batch_size=batch_size, num_workers=num_workers)
        self.validation_dataloader = DataLoader(validation_dataset, batch_size=1, num_workers=num_workers)
        pass

    @abc.abstractmethod
    def train_epoch(self) -> None:
        """Abstrat method that will implement the training loop strategy for one dataset epoch
        """
        pass

    @abc.abstractmethod
    def validation_epoch(self) -> None:
        """Abstrat method that will implement the training loop strategy for one dataset epoch
        """
        pass

    @abc.abstractmethod
    def train_model(self):
        """Abstract method that implements the training logic over several epochs
        """
        pass

    def save_model(self, save_dir: str = None):
        """Save the model in the selected directory. If None is passed, it is saved in the default directory specified 
        in model options.

        Args:
            save_dir (str, optional). Defaults to None.
        """

        self.model.save_model(save_dir)
