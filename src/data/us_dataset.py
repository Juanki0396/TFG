from typing import Tuple, List, Dict

import torch
from torch.utils.data import Dataset

from ..data.image import Ultrasound
from .data_transforms import Compose, Transform


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
        tensor = self.transforms(us.torch_tensor).float().unsqueeze(0)
        label = torch.tensor(self.label_dict[us.label], dtype=torch.int64).unsqueeze(0)

        return tensor, label
