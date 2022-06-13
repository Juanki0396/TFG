from argparse import Namespace
from itertools import chain
from typing import Dict, List, Tuple

import torch

from . import networks
from .base_model import BaseModel
from ..options.cyclegan_options import CycleGanOptions
from ..data.image import Image


class CycleGan(BaseModel):

    @staticmethod
    def set_scheduler(optimizer: torch.optim.Optimizer, options: Namespace) -> torch.optim.lr_scheduler.LambdaLR:

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - options.epochs_constant + 1) / float(options.epochs_decay + 1)
            return lr_l

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        return scheduler

    def __init__(self, parser: CycleGanOptions) -> None:

        super().__init__(parser=parser)

        if self.options.net_G == "unet":
            self.models["net_GA"] = networks.UnetGenerator(3, 3, 5).to(self.device)
            self.models["net_GB"] = networks.UnetGenerator(3, 3, 5).to(self.device)
        elif self.options.net_G == "resnet":
            self.models["net_GA"] = networks.ResnetGenerator(3, 3).to(self.device)
            self.models["net_GB"] = networks.ResnetGenerator(3, 3).to(self.device)
        else:
            raise NotImplementedError(f"Generetor model {self.options.net_G} is not implemented.")

        if self.options.net_D == "patch":
            self.models["net_DA"] = networks.NLayerDiscriminator(3).to(self.device)
            self.models["net_DB"] = networks.NLayerDiscriminator(3).to(self.device)
        elif self.options.net_D == "pixel":
            self.models["net_DA"] = networks.PixelDiscriminator(3).to(self.device)
            self.models["net_DB"] = networks.PixelDiscriminator(3).to(self.device)
        else:
            raise NotImplementedError(f"Discriminator model {self.options.net_D} is not implemented.")

        self.criterions["gan"] = torch.nn.BCEWithLogitsLoss()
        self.criterions["cycle"] = torch.nn.L1Loss()

        self.optimizers["G"] = torch.optim.Adam(
            chain(self.models["net_GA"].parameters(), self.models["net_GB"].parameters()),
            lr=self.options.learning_rate, betas=(self.options.beta, 0.999)
        )
        self.optimizers["D"] = torch.optim.Adam(
            chain(self.models["net_DA"].parameters(), self.models["net_DB"].parameters()),
            lr=self.options.learning_rate, betas=(self.options.beta, 0.999)
        )

        if not self.options.lr_constant:

            self.schedulers = {}
            self.schedulers["G"] = self.set_scheduler(self.optimizers["G"], self.options)
            self.schedulers["D"] = self.set_scheduler(self.optimizers["D"], self.options)

        else:

            self.schedulers = None

    def set_input(self, input: Dict[str, torch.Tensor]) -> None:

        self.real_A: torch.Tensor = input["A"].to(self.device)
        self.real_B: torch.Tensor = input["B"].to(self.device)

    def forward(self) -> None:

        self.fake_B: torch.Tensor = self.models["net_GA"](self.real_A)
        self.rec_A: torch.Tensor = self.models["net_GB"](self.fake_B)
        self.fake_A: torch.Tensor = self.models["net_GB"](self.real_B)
        self.rec_B: torch.Tensor = self.models["net_GA"](self.fake_A)

    def basic_backward_D(self, net_D: torch.nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:

        pred_real = net_D(real)
        loss_real = self.criterions["gan"](pred_real, torch.Tensor([1.0]).expand_as(pred_real).to(self.device))

        pred_fake = net_D(fake.detach())
        loss_fake = self.criterions["gan"](pred_fake, torch.Tensor([0.0]).expand_as(pred_fake).to(self.device))

        total_loss = (loss_fake + loss_real) * 0.5
        total_loss.backward()
        return total_loss.item()

    def backward_D(self) -> Dict[str, float]:

        total_loss_DA = self.basic_backward_D(self.models["net_DA"], self.real_B, self.fake_B)
        total_loss_DB = self.basic_backward_D(self.models["net_DB"], self.real_A, self.fake_A)
        return {"loss_DA": total_loss_DA, "loss_DB": total_loss_DB}

    def backward_G(self) -> Dict[str, float]:

        lambda_ = 10

        pred_B = self.models["net_DA"](self.fake_B)
        pred_A = self.models["net_DB"](self.fake_A)
        loss_GA = self.criterions["gan"](pred_B, torch.Tensor([1.0]).expand_as(pred_B).to(self.device))
        loss_GB = self.criterions["gan"](pred_A, torch.Tensor([1.0]).expand_as(pred_A).to(self.device))
        loss_cycleA = self.criterions["cycle"](self.rec_A, self.real_A) * lambda_
        loss_cycleB = self.criterions["cycle"](self.rec_B, self.real_B) * lambda_
        total_loss = loss_GA + loss_GB + loss_cycleA + loss_cycleB
        total_loss.backward()

        return {"loss_GA": loss_GA.item(), "loss_GB": loss_GB.item(), "loss_cycle": (loss_cycleA + loss_cycleB).item()}

    def update_parameters(self) -> Dict[str, float]:

        super().update_parameters()

        self.set_requires_grad([self.models["net_DA"], self.models["net_DB"]], False)
        self.optimizers["G"].zero_grad()
        loss_G = self.backward_G()
        self.optimizers["G"].step()

        self.set_requires_grad([self.models["net_DA"], self.models["net_DB"]], True)
        self.optimizers["D"].zero_grad()
        loss_D = self.backward_D()
        self.optimizers["D"].step()

        if self.schedulers is not None:
            self.schedulers["G"].step()
            self.schedulers["D"].step()

        loss = {**loss_D, **loss_G}

        return loss

    def validation(self) -> Tuple[List[Image], Dict[str, float]]:

        super().validation()

        with torch.no_grad():
            self.forward()
            metric = self.compute_metric()

        images = [
            Image(self.fake_B.squeeze().cpu().detach(), "Fake_B"),
            Image(self.fake_A.squeeze().cpu().detach(), "Fake_A"),
            Image(self.rec_A.squeeze().cpu().detach(), "Cycle_A"),
            Image(self.rec_B.squeeze().cpu().detach(), "Cycle_B")
        ]

        return images, metric

    def compute_metric(self):
        pass

    def inference(self, input: Dict[str, torch.Tensor] = None) -> List[Image]:

        super().inference(input)

        fake_B: torch.Tensor = self.models["net_GA"](input["A"].to(self.device))
        fake_A: torch.Tensor = self.models["net_GB"](input["B"].to(self.device))

        images = [
            Image(fake_B.squeeze().cpu().detach(), "Fake_B"),
            Image(fake_A.squeeze().cpu().detach(), "Fake_B")
        ]

        return images
