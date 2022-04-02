import torch
from torchvision.models import resnet18


def resnet18_classifier(n_labels: int) -> torch.nn.Module:
    model = resnet18(False, False)
    model.fc = torch.nn.Linear(in_features=512, out_features=n_labels)
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(
                layer.weight, mode="fan_out", nonlinearity="relu")

        if isinstance(layer, torch.nn.BatchNorm2d):
            torch.nn.init.constant(layer.weight, 1)
            torch.nn.init.constant(layer.bias, 0)
    return model


if __name__ == "__main__":
    print(resnet18_classifier(2))
    pass
