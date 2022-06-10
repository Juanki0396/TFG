

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
            img = torch.from_numpy(img).double()

        label = label.unsqueeze(dim=0)

        return img.float(), label


class CycleGanDataset(Dataset):

    def __init__(self, dataset: List[Image], transforms: List[Transform] = None) -> None:

        self.dataset_A = list(filter(lambda image: image.label == "A", dataset))
        self.dataset_B = list(filter(lambda image: image.label == "B", dataset))
        self.transform = Compose(transforms=transforms)

    def shuffle(self) -> None:

        random.shuffle(self.dataset_A)
        random.shuffle(self.dataset_B)

    def __len__(self) -> int:
        return min(len(self.dataset_A), len(self.dataset_B))

    def __getitem__(self, index) -> Dict[str, np.ndarray]:

        data = {}

        for img in [self.dataset_A[index], self.dataset_B[index]]:
            tensor = img.torch_tensor.repeat(3, 1, 1)
            tensor = self.transform(tensor)
            data[img.label] = tensor

        return data


if __name__ == "__main__":

    train_data_path = "/home/quito/Fisica/tfg/Data/prototype/train_SERAM.npy"

    dataset = ImageDataset(train_data_path, random_labels=True)

    img, label = dataset[0]
    print(img.shape)
