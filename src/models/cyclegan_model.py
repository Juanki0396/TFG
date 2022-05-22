
from itertools import chain
from typing import Dict

import torch

from . import networks
from .base_model import BaseModel


class CycleGan(BaseModel):

    def __init__(self, net_G: str = "resnet", net_D: str = "patch", device: str = "cpu", lr: float = 1e-3):

        super().__init__()

        self.device = device
        self.name = f"cyclegan_{net_G}_{net_D}_lr_{lr:.2e}"

        if net_G == "unet":
            self.models["net_GA"] = networks.UnetGenerator(3, 3, 5).to(device)
            self.models["net_GB"] = networks.UnetGenerator(3, 3, 5).to(device)
        elif net_G == "resnet":
            self.models["net_GA"] = networks.ResnetGenerator(3, 3).to(device)
            self.models["net_GB"] = networks.ResnetGenerator(3, 3).to(device)
        else:
            raise NotImplementedError(f"Generetor model {net_G} is not implemented.")

        if net_D == "patch":
            self.models["net_DA"] = networks.NLayerDiscriminator(3).to(device)
            self.models["net_DB"] = networks.NLayerDiscriminator(3).to(device)
        elif net_D == "pixel":
            self.models["net_DA"] = networks.PixelDiscriminator(3).to(device)
            self.models["net_DB"] = networks.PixelDiscriminator(3).to(device)
        else:
            raise NotImplementedError(f"Discriminator model {net_D} is not implemented.")

        self.criterions["gan"] = torch.nn.BCEWithLogitsLoss()
        self.criterions["cycle"] = torch.nn.L1Loss()

        self.optimizers["G"] = torch.optim.Adam(
            chain(self.models["net_GA"].parameters(), self.models["net_GB"].parameters()),
            lr=lr
        )
        self.optimizers["D"] = torch.optim.Adam(
            chain(self.models["net_DA"].parameters(), self.models["net_DB"].parameters()),
            lr=lr
        )

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

        self.forward()

        self.set_requires_grad([self.models["net_DA"], self.models["net_DB"]], False)
        self.optimizers["G"].zero_grad()
        loss_G = self.backward_G()
        self.optimizers["G"].step()

        self.set_requires_grad([self.models["net_DA"], self.models["net_DB"]], True)
        self.optimizers["D"].zero_grad()
        loss_D = self.backward_D()
        self.optimizers["D"].step()

        loss_G.update(loss_D)

        return loss_G

    def test(self) -> Dict[str, torch.Tensor]:

        self.set_train_mode(False)
        with torch.no_grad():
            self.forward()
            metric = self.compute_metric()

        results = {
            "real_A": self.real_A.detach(),
            "fake_A": self.fake_A.detach(),
            "rec_A": self.rec_A.detach(),
            "real_B": self.real_B.detach(),
            "fake_B": self.fake_A.detach(),
            "rec_B": self.rec_A.detach(),
        }

        return results

    def compute_metric(self):
        pass