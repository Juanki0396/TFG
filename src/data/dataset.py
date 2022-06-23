

import random
from typing import Dict, Tuple, List

import torch
import numpy as np
from torch.utils.data import Dataset

from .image import Image
from .data_transforms import RandomNoise, DynamicRangeScaling, Resize, Compose, Transform
from ..options.classifier_options import ClassifierOptions


def generate_noisy_dataset(data: List[Image], std: float) -> List[Image]:
    """Takes a list of images and add noise to half of them with a normal distribution
    of mean 0 and the specified std. The transformed images are labeled with a 1 and the
    others with o.

    Args:
        data (List[Image])
        std (float)

    Returns:
        List[Image]
    """

    labels = [1 if i <= (len(data)//2) else 0 for i in range(len(data))]
    random.shuffle(labels)
    random_transform = RandomNoise(0, std)
    new_data = []

    for i, label in enumerate(labels):
        img = data[i]

        if label == 1:
            img.image = random_transform(img.image)
            img.label = label

        else:
            img.label = label

        new_data.append(img)

    return new_data


def generate_cyclegan_dataset(data: List[Image], std: float) -> List[Image]:
    """Generate a cyclegan dataset where dataset A are normal images and dataset
    B are noisy images.

    Args:
        data (List[Image])
        std (float): Standard deviation of the added noise.

    Returns:
        List[Image]
    """
    new_data = generate_noisy_dataset(data, std)
    for image in new_data:
        if image.label == 1:
            image.label = "B"
        elif image.label == 0:
            image.label = "A"
        else:
            raise ValueError(f"Image label is not 1 or 0 -> {image.label}")

    return new_data


class ImageDataset(Dataset):

    def __init__(self, dataset: List[Image], transforms: List[Transform] = None) -> None:

        super().__init__()

        self.dataset = dataset
        self.transform = Compose(transforms=transforms)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:

        data = self.dataset[index]
        img, label = data.torch_tensor, torch.tensor(data.label).float()
        img = self.transform(img)

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()

        label = label.unsqueeze(dim=0)

        return img, label
