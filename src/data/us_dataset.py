import random
from typing import Tuple, List, Dict

import torch
from torch.utils.data import Dataset

from ..data.image import Ultrasound
from .data_transforms import Compose, Transform


def split_us_dataset(us_dataset: List[Ultrasound], labels: List[str], validation_fraction: float, seed: int = 1234) -> Tuple[List[Ultrasound], ...]:
    """Filters the US dataset given by the label and divide it into two datasets that contains the specified fraction for each label.
    If a label is not present in labels, it will be ignored.

    Args:
        data (List[Ultrasound]): dataset to split
        labels (List[str]): labels to filter in the dataset
        validation_fraction (float): Which fraction of the images will be in the validation dataset

    Returns:
        Tuple[List[Ultrasound],...]: Train dataset and Validation dataset
    """

    if not 0 <= validation_fraction < 1:
        raise ValueError(f"Validation fraction should be a number between 0 and 1 -> {validation_fraction} does not fulfill the condition")

    random.seed(seed)

    train_dataset = []
    validation_dataset = []

    for label in labels:
        label_data = [us for us in filter(lambda us: us.label == label, us_dataset)]
        n_validation = int(len(label_data) * validation_fraction)
        train_dataset.extend(label_data[n_validation:])
        validation_dataset.extend(label_data[:n_validation])

    random.shuffle(train_dataset)
    random.shuffle(validation_dataset)

    return train_dataset, validation_dataset


class UsDataset(Dataset):

    def __init__(self, us_list: List[Ultrasound], label_dict: Dict[str, int], transforms: List[Transform] = None, ) -> None:

        self.dataset = us_list
        self.transforms = Compose(transforms=transforms)
        self.label_dict = label_dict

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> Tuple[torch.Tensor, ...]:

        us = self.dataset[index]
        us.dark_cone()
        tensor = self.transforms(us.torch_tensor).to(torch.float32)
        label = torch.tensor(self.label_dict[us.label]).to(torch.int64)

        return tensor, label
