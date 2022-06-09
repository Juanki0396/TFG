
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from torch.utils.data import Dataset


from ..utils import run_time
from ..models.base_trainer import BaseTrainer
from ..models.clasifier_model import ImageClassifier


class ClassifierTrainer(BaseTrainer):

    def __init__(self, model: ImageClassifier) -> None:
        """Instanciate a ClassifierTrainer by passing the ImageClassifier model to train.

        Args:
            model (ImageClassifier)
        """

        super().__init__(model)

        self.model: ImageClassifier

        self.training_loss = []
        self.validation_loss = []
        self.metric = []
        self.best_metric = 0.

    def set_dataloaders(self, training_dataset: Dataset, validation_dataset: Dataset) -> None:
        return super().set_dataloaders(training_dataset, validation_dataset)

    @run_time
    def train_epoch(self):
        """Run a training pass over the whole training dataset and store the batch losses.
        """

        for imgs, labels in self.traning_dataloader:

            input = {"images": imgs, "labels": labels}
            self.model.set_input(input)
            loss = self.model.update_parameters()
            self.training_loss.append[loss]

            del imgs, labels

    @run_time
    def validation_epoch(self):
        """Run a vladiation step over the whole validating dataset and store the bath losses and metrics.
        """

        for imgs, labels in self.validation_dataloader:

            self.model.set_input(imgs, labels)
            loss, metric = self.model.validation()
            self.validation_loss.append(loss)
            self.metric.append(metric)

            del imgs, labels

    @run_time
    def train_model(self, epochs: int = None):
        """Run training and validation steps for a certain number of epochs. Each time a best metric is
        achived, the model is saved

        Args:
            epochs (int)
        """
        if epochs is None:
            epochs = self.model.options.epochs

        for epoch in range(epochs):
            print(f"Training Epoch: {epoch}".center(60, "-"))
            self.train_epoch()
            self.validation_epoch()

            epoch_metric = np.mean(self.metric[-len(self.validation_dataloader):])
            print(f"Epoch metric: {epoch_metric:.3f}")

            if epoch_metric > self.best_metric:
                self.save_model()
                self.best_metric = epoch_metric
                print(f"New model saved")

    def plot_losses(self) -> Figure:
        """Plot the epoch losses and metrics

        Returns:
            Figure
        """

        epoch_training_loss = np.array(self.training_loss).reshape((-1, len(self.traning_dataloader))).mean(axis=1)
        epoch_validation_loss = np.array(self.validation_loss).reshape((-1, len(self.validation_dataloader))).mean(axis=1)
        epoch_metric = np.array(self.metric).reshape((-1, len(self.validation_dataloader))).mean(axis=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.plot(epoch_training_loss, label="Train")
        ax1.plot(epoch_validation_loss, label="Validation")
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel(self.model.options.loss_function)
        ax1.set_title("Loss")
        ax1.legend()

        ax2.plot(epoch_metric)
        ax2.set_ylim(bottom=0)
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel(self.model.options.metric)
        ax2.set_title("Metric")

        return fig
