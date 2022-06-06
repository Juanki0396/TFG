
import os
import random
from typing import Dict, Tuple, List
from collections import namedtuple


import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Normalize, Compose

from .image import Image
from .data_transforms import RandomNoise, DynamicRangeScaling


def generate_noisy_dataset(data: List[Image]):

    new_data = []

    labels = [1 if i <= (len(data)//2) else 0 for i in range(len(data))]
    random.shuffle(labels)

    random_transform = RandomNoise(0, 20)

    for i, label in enumerate(labels):
        img = data[i]
        if label == 1:

            img.image = random_transform(img.image)
            img.label = label
        else:
            img.label = label
        new_data.append(img)
    return new_data


def generate_cycle_gan_dataset(data):

    noisy_dataset = generate_noisy_dataset(data)
    images_A = filter(lambda x: x[1] == 0, noisy_dataset)
    images_B = filter(lambda x: x[1] == 1, noisy_dataset)
    cyclegan_dataset = []
    for tupA, tupB in zip(images_A, images_B):
        data = {"A": tupA[0], "B": tupB[0]}
        cyclegan_dataset.append(data)
    return cyclegan_dataset


class ImageDataset(Dataset):

    def __init__(
            self, dataset: List[Image]) -> None:

        super().__init__()

        self.dataset = dataset
        self.transform = DynamicRangeScaling()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:

        data = self.dataset[index]
        img, label = data.torch_tensor, torch.tensor(data.label)
        img = self.transform(img)

        return img, label


class CycleGanDataset(Dataset):

    def __init__(self, dataset: List[Dict[str, np.ndarray]], image_output_size: Tuple[int, int] = (256, 256)) -> None:

        self.dataset = dataset
        self.__image_size = image_output_size
        self.__transform = Compose([
            Resize(self.__image_size),
            DynamicRangeScaling()
        ])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> Dict[str, np.ndarray]:

        data = {}
        for key, x in self.dataset[index].items():
            img = torch.from_numpy(np.squeeze(x.copy()).astype(np.float32)).unsqueeze(0).repeat(3, 1, 1)
            img = self.__transform(img)
            data[key] = img

        return data


if __name__ == "__main__":

    train_data_path = "/home/quito/Fisica/tfg/Data/prototype/train_SERAM.npy"

    dataset = ImageDataset(train_data_path, random_labels=True)

    img, label = dataset[0]
    print(img.shape)
