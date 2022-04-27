
import abc
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils import run_time
from src.models.base_model import BaseModel, CycleGan, ImageClassifier


class BaseTrainer(abc.ABC):

    def __init__(self) -> None:
        super().__init__()

        self.model = None

    def set_dataloaders(self, train_dataloader: DataLoader, validation_dataloader: DataLoader) -> None:

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

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
        pass


class ClassifierTrainer(BaseTrainer):

    def __init__(self, model: ImageClassifier) -> None:
        super().__init__()

        self.model = model
        self.epoch_train_loss = []
        self.epoch_validation_loss = []
        self.epoch_metric = []

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
