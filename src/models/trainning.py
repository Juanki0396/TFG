
import abc
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


from src.utils import run_time
from src.models.base_model import BaseModel, CycleGan, ImageClassifier


class BaseTrainer(abc.ABC):

    def __init__(self) -> None:
        super().__init__()

        self.model = None

    @abc.abstractmethod
    def set_dataloaders(self) -> None:
        pass

    @abc.abstractmethod
    def train_epoch(self):
        pass

    @abc.abstractmethod
    def validation_epoch(self):
        pass


class CycleGanTrainer(BaseTrainer):

    def __init__(self, model: CycleGan) -> None:
        super().__init__()

        self.model = model
        self.train_losss = []

    def set_dataloaders(self, train_dataloader: DataLoader) -> None:

        self.train_dataloader = train_dataloader

    @run_time
    def train_epoch(self):

        self.model.set_train_mode(True)

        for input in self.train_dataloader:

            self.model.set_input(input)
            batch_losses = self.model.update_parameters()
            self.train_losss.append(batch_losses)

    @run_time
    def validation_epoch(self):

        self.model.set_train_mode(False)
        self.validation_images = []

        for batch, input in enumerate(self.validation_dataloader):
            self.model.set_input(input)
            images = self.model.test()
            if batch % 20 == 0:
                self.validation_images.append(images)

    def plot_training_losses(self):

        epochs = range(len(self.train_losss))
        loss_DA = map(lambda x: x["loss_DA"], self.train_losss)
        loss_DB = map(lambda x: x["loss_DB"], self.train_losss)
        loss_GA = map(lambda x: x["loss_GA"], self.train_losss)
        loss_GB = map(lambda x: x["loss_GB"], self.train_losss)
        #loss_cycle = map(lambda x: x["loss_cycle"], self.train_losss)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.plot(epochs, list(loss_GA), label="Generator A")
        ax1.plot(epochs, list(loss_GB), label="Generator B")
        #ax1.set_ylim(0, 2)
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("BCE with Logistic")
        ax1.set_title("Generator losses")
        ax1.legend()

        ax2.plot(epochs, list(loss_DA), label="Discriminator A")
        ax2.plot(epochs, list(loss_DB), label="Discriminator B")
        #ax2.set_ylim(0, 1)
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("BCE with Logistic")
        ax2.set_title("Discriminators")

        return fig


class ClassifierTrainer(BaseTrainer):

    def __init__(self, model: ImageClassifier, save_dir: str) -> None:
        super().__init__()

        self.model = model
        self.save_dir = save_dir
        self.epoch_train_loss = []
        self.epoch_validation_loss = []
        self.epoch_metric = []
        self.metric_model_saved = 0.
        self.train_loss_model_saved = None

    def set_dataloaders(self, train_dataloader: DataLoader, validation_dataloader: DataLoader) -> None:

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

    @run_time
    def train_epoch(self):

        self.model.set_train_mode(True)
        batch_losses = []

        for imgs, labels in self.train_dataloader:

            self.model.set_input(imgs, labels)
            losses = self.model.update_parameters()
            batch_losses.append(losses["BCEloss"])

            del imgs, labels

        mean_loss = np.mean(batch_losses)
        self.epoch_train_loss.append(mean_loss)

    @run_time
    def validation_epoch(self):

        self.model.set_train_mode(False)
        batch_losses = []
        batch_metrics = []

        for imgs, labels in self.validation_dataloader:

            self.model.set_input(imgs, labels)
            losses = self.model.test()
            batch_losses.append(losses["BCEloss"])
            batch_metrics.append(losses["Accuracy"])

            del imgs, labels

        mean_loss = np.mean(batch_losses)
        mean_metric = np.mean(batch_metrics)
        self.epoch_validation_loss.append(mean_loss)
        self.epoch_metric.append(mean_metric)

    def save_model(self):

        if self.epoch_metric[-1] > self.metric_model_saved:
            self.model.save_model(self.save_dir)
            self.metric_model_saved = self.epoch_metric[-1]
            self.train_loss_model_saved = self.epoch_train_loss[-1]
            print(f"Best metric achieved: {self.metric_model_saved} -> Model saved")

    def plot_losses(self):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        epochs = range(len(self.epoch_train_loss))

        ax1.plot(epochs, self.epoch_train_loss, label="Train")
        ax1.plot(epochs, self.epoch_validation_loss, label="Validation")
        ax1.set_ylim(0, 2)
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("BCELoss")
        ax1.set_title("Losses")
        ax1.legend()

        ax2.plot(epochs, self.epoch_metric)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Metrics")

        return fig


def train_classifiers(train_manager: ClassifierTrainer, epochs: int) -> ClassifierTrainer:
    for epoch in range(epochs):
        print(f"Epoch {epoch}".center(30, "-"))
        print(" Starting training loop")
        train_manager.train_epoch()
        print(" Starting validation loop")
        train_manager.validation_epoch()
        print(f"Results")
        print(f"Train loss: {train_manager.epoch_train_loss[epoch]}")
        print(f"Validation loss: {train_manager.epoch_validation_loss[epoch]}")
        print(f"Metric: {train_manager.epoch_metric[epoch]}")
        train_manager.save_model()

    return train_manager


def train_cyclegan(train_manager: CycleGanTrainer, epochs: int) -> CycleGanTrainer:
    for epoch in range(epochs):
        print(f"Epoch {epoch}".center(30, "-"))
        print(" Starting training loop")
        train_manager.train_epoch()
        print(f"Results")
        print(f"Losses:\n{train_manager.train_losss[-1]}")
        train_manager.save_model()

    return train_manager
