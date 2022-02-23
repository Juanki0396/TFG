
from numpy import identity
import torch
import torch.nn as nn
from torchvision.models import resnet18
from typing import Type, Any, Callable, Union, List, Optional


def conv3x3(in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        """Creates an instance of the basic block of a ResNet. This nets are made repeting these blocks

        Args:
            in_channels (int): input number of channels.
            out_channels (int): output number of channels
            stride (int, optional): Stride applied by the 1st conv layer. Defaults to 1.
            downsample (Optional[nn.Module], optional): downsampling method to apply for the residual conection
            when stride is greater than 1. Defaults to None.
        """
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x


def create_resnet():
    return resnet18()


if __name__ == "__main__":

    x = torch.zeros((1, 3, 28, 28))

    layer1 = conv3x3(3, 6, 2)
    layer2 = conv1x1(3, 6, 2)

    y1 = layer1(x)
    y2 = layer2(x)

    print(f"3x3 -> {y1.shape}")
    print(f"1x1 -> {y2.shape}")
