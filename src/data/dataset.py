
import os
import random
from typing import Tuple, List


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


class ImageDataset(Dataset):

    def __init__(
            self, dataset: List[Tuple[np.ndarray, int]], image_output_size: Tuple[int, int] = (256, 256),
            random_labels: bool = False) -> None:
        super().__init__()
        self.__data = dataset
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
        return len(self.__data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        img, label = self.__data[index]
        img = torch.from_numpy(np.squeeze(img).astype(np.float32)).unsqueeze(0).repeat(3, 1, 1)
        img = self.__transform(img)
        if len(self.__random_labels) != 0:
            label = self.__random_labels[index]

        return img, label


if __name__ == "__main__":

    train_data_path = "/home/quito/Fisica/tfg/Data/prototype/train_SERAM.npy"

    dataset = ImageDataset(train_data_path, random_labels=True)

    img, label = dataset[0]
    print(img.shape)
