
from cProfile import label
import os
import random
from typing import Tuple


import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Normalize, Compose


def Normalization(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


class ImageDataset(Dataset):

    def __init__(
            self, dataset_path: str, image_output_size: Tuple[int, int] = (256, 256),
            random_labels: bool = False) -> None:
        super().__init__()
        self.__data = np.load(dataset_path, allow_pickle=True)
        self.__image_size = image_output_size
        self.__transform = Compose([
            Resize(self.__image_size),
            Normalization
        ])

        if random_labels:
            self.__random_labels = [random.randint(0, 1) for _ in range(len(self))]
        else:
            self.__random_labels = []

    def __len__(self) -> int:
        return self.__data.shape[0]

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        img, label = self.__data[index]
        img = torch.from_numpy(np.squeeze(img).astype(np.float32)).unsqueeze(0).repeat(3, 1, 1)
        img = self.__transform(img)
        if len(self.__random_labels) != 0:
            label = random.randint(0, 1)

        return img, label


if __name__ == "__main__":

    train_data_path = "/home/quito/Fisica/tfg/Data/prototype/train_SERAM.npy"

    dataset = ImageDataset(train_data_path, random_labels=True)

    img, label = dataset[0]
    print(img.shape)
