
import os
import random
from typing import Dict, Tuple, List
from collections import namedtuple


import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Normalize, Compose

from config import SEED

random.seed(SEED)
np.random.seed(SEED)


def load_data(numpy_path: str) -> List[Tuple[np.ndarray, int]]:
    array = np.load(numpy_path, allow_pickle=True)
    data = [(img, label) for img, label in array]
    return data


def add_random_noise(img: np.ndarray) -> np.ndarray:
    noise = np.random.normal(img.mean(), img.std(), img.shape)
    return noise + img


def Normalization(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def generate_noisy_dataset(data):

    new_data = []
    labels = [1 if i <= (len(data)//2) else 0 for i in range(len(data))]
    random.shuffle(labels)
    for i, label in enumerate(labels):
        if label == 1:
            img = data[i][0]
            x = add_random_noise(img)
            y = label
        else:
            x = data[i][0]
            y = label
        new_data.append((x, y))
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
            self, dataset: List[Tuple[np.ndarray, int]], image_output_size: Tuple[int, int] = (256, 256),
            random_labels: bool = False) -> None:
        super().__init__()
        self.__dataset = dataset
        self.__image_size = image_output_size
        self.__transform = Compose([
            Resize(self.__image_size),
            Normalization
        ])

        if random_labels:
            self.__random_labels = [1 if i <= len(self)//2 else 0 for i in range(len(self))]
            random.shuffle(self.__random_labels)
        else:
            self.__random_labels = []

    def __len__(self) -> int:
        return len(self.__dataset)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        img, label = self.__dataset[index]
        img = torch.from_numpy(np.squeeze(img).astype(np.float32)).unsqueeze(0).repeat(3, 1, 1)
        img = self.__transform(img)
        if len(self.__random_labels) != 0:
            label = self.__random_labels[index]

        return img, torch.FloatTensor([label])


class CycleGanDataset(Dataset):

    def __init__(self, dataset: List[Dict[str, np.ndarray]], image_output_size: Tuple[int, int] = (256, 256)) -> None:

        self.dataset = dataset
        self.__image_size = image_output_size
        self.__transform = Compose([
            Resize(self.__image_size),
            Normalization
        ])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> Dict[str, np.ndarray]:

        data = self.dataset[index]
        for key, img in data.items():
            img = torch.from_numpy(np.squeeze(img).astype(np.float32)).unsqueeze(0).repeat(3, 1, 1)
            img = self.__transform(img)
            data[key] = img

        return data


if __name__ == "__main__":

    train_data_path = "/home/quito/Fisica/tfg/Data/prototype/train_SERAM.npy"

    dataset = ImageDataset(train_data_path, random_labels=True)

    img, label = dataset[0]
    print(img.shape)
