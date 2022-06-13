
from calendar import EPOCH
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset


from ..utils import run_time
from ..data.image import Image
from ..models.base_trainer import BaseTrainer
from ..models.cyclegan_model import CycleGan


class CycleGanTrainer(BaseTrainer):

    def __init__(self, model: CycleGan) -> None:
        super().__init__(model)

        self.model: CycleGan
        self.losses = []
        self.model_saved_loss = {"loss_GA": np.inf, "loss_GB": np.inf}

    def set_dataloaders(self, training_dataset: Dataset, validation_dataset: Dataset) -> None:
        return super().set_dataloaders(training_dataset, validation_dataset)

    @run_time
    def train_epoch(self):

        for i, input in enumerate(self.traning_dataloader):

            print(f"\r Running training batch {i+1}/{len(self.traning_dataloader)}", end="")
            self.model.set_input(input)
            batch_losses = self.model.update_parameters()
            self.losses.append(batch_losses)

        print("\nTraining epoch fisnished succesfully")

    @run_time
    def validation_epoch(self):

        print("Running validation epoch")

        for input in self.validation_dataloader:
            self.model.set_input(input)
            images, _ = self.model.validation()
            fig = Image.create_image_grid(images, grid_size=(14, 10))

        print("Validation epoch fisnished succesfully")

    @run_time
    def train_model(self):

        epochs = self.model.options.epochs_constant + self.model.options.epochs_decay

        for epoch in range(epochs):

            print(f"EPOCH {epoch}".center(60, "-"))

            self.train_epoch()

            loss_GA = np.mean([batch_loss["loss_GA"] for batch_loss in self.losses[-len(self.traning_dataloader):]])
            loss_GB = np.mean([batch_loss["loss_GB"] for batch_loss in self.losses[-len(self.traning_dataloader):]])

            if loss_GA < self.model_saved_loss["loss_GA"] and loss_GB < self.model_saved_loss["loss_GB"]:
                self.save_model()
                self.model_saved_loss["loss_GA"] = loss_GA
                self.model_saved_loss["loss_GB"] = loss_GB
                print(f"New model saved with metrics -> loss_GA: {loss_GA}  ;  loss_GB: {loss_GB}")

        self.validation_epoch()

    def plot_training_losses(self):

        batchs_per_epoch = len(self.traning_dataloader)
        loss_DA = np.array(list(map(lambda x: x["loss_DA"], self.losses))).reshape((-1, batchs_per_epoch)).mean(axis=1)
        loss_DB = np.array(list(map(lambda x: x["loss_DB"], self.losses))).reshape((-1, batchs_per_epoch)).mean(axis=1)
        loss_GA = np.array(list(map(lambda x: x["loss_GA"], self.losses))).reshape((-1, batchs_per_epoch)).mean(axis=1)
        loss_GB = np.array(list(map(lambda x: x["loss_GB"], self.losses))).reshape((-1, batchs_per_epoch)).mean(axis=1)
        #loss_cycle = map(lambda x: x["loss_cycle"], self.train_losss)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.plot(loss_GA, label="Generator A")
        ax1.plot(loss_GB, label="Generator B")
        #ax1.set_ylim(0, 2)
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("BCE with Logistic")
        ax1.set_title("Generator losses")
        ax1.legend()

        ax2.plot(loss_DA, label="Discriminator A")
        ax2.plot(loss_DB, label="Discriminator B")
        #ax2.set_ylim(0, 1)
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("BCE with Logistic")
        ax2.set_title("Discriminators")

        return fig
