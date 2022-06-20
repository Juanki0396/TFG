
import datetime
import os
import random
import time
from typing import Callable, Any, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from .data.image import Ultrasound, Image


def run_time(f: Callable) -> Callable:
    """Decorator that meassures the execution time of a function.

    Args:
        f (Callable): Function to decorate.
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t = time.time()
        result = f(*args, **kwargs)
        t = time.time() - t
        print(f"Running time -> {datetime.timedelta(seconds=t//1)}")
        return result
    return wrapper


def load_xray_data(dataset_path: str = "Data/x_ray") -> List[Image]:
    """Load Chest x rays dataset.

    Args:
        dataset_path (str, optional): Defaults to "Data/x_ray".

    Returns:
        List[Image]
    """
    train_path = os.path.join(dataset_path, "train_SERAM.npy")
    test_path = os.path.join(dataset_path, "test1_SERAM.npy")

    data = []

    for path in [train_path, test_path]:
        array = np.load(path, allow_pickle=True)
        data = data + [Image(img, label) for img, label in array]

    return data


def load_us_data(dataset_path: str = "Data/USAnotAI") -> List[Ultrasound]:

    files = [file for file in os.scandir(dataset_path) if file.name not in ("source.md", "cone.png")]
    cone = Image.from_path(os.path.join(dataset_path, "cone.png"), "cone")
    data = [Image.from_path(file.path, file.name.split("-")[0]) for file in files]
    data = [Ultrasound(img.image, img.label, cone.image) for img in data]

    return data


def refactor_dataset(data: List[Image], ratio_validation: float = 0.2) -> Tuple[List[Image], ...]:
    """Divide data into training and validation datasets with the selected ratios.

    Args:
        data (List[Image]): 
        ratio_validation (float, optional): . Defaults to 0.2.

    Returns:
        Tuple[List[Image], ...]: training and validation datsets
    """
    random.shuffle(data)
    n_validation = int(len(data) * ratio_validation)
    training_data = data[n_validation:]
    validation_data = data[:n_validation]

    return training_data, validation_data


def refactor_cyclegan_datasets(data: List[Image], n_validation: int = 10) -> Tuple[List[Image], ...]:
    """Obtain the training and validation datasets for cyclegan from data list. Cyclegan
    only needs validation dataset for visual inspection, so only a few images are need it.

    Args:
        data (List[Image])
        n_validation (int, optional). Defaults to 10.

    Returns:
        Tuple[List[Image], ...]
    """
    random.shuffle(data)
    training_data = data[n_validation:]
    validation_data = data[:n_validation]

    return training_data, validation_data


def hist_integral(y: List[float], x: List[float]) -> float:
    """Compute the integral given the heights and bins of the histogram.

    Args:
        y (List[float]): Heights of the bins
        x (List[float]): Limits of the bins

    Returns:
        float: Integral for different x_f
    """
    y = np.array(y)
    dx = np.diff(x)
    area = np.cumsum(y * dx)
    return area


def obtain_voxel_range(data: List[Image], sample_size: int, extreme_percent: int = 2) -> Tuple[float, float]:
    """Obtain the limit values of voxels which exclude the percentage of the data defined by the user.

    Args:
        data (List[Image])
        sample_size (int)
        extreme_percent (int, optional): Defaults to 2.

    Returns:
        Tuple[float, float]
    """

    sample = [img.image.flatten() for img in random.sample(data, sample_size)]
    voxels = np.concatenate(sample)

    min_voxel = np.percentile(voxels, extreme_percent//2)
    max_voxel = np.percentile(voxels, 100 - extreme_percent//2)
    voxel_range = (min_voxel, max_voxel)

    return voxel_range


def obtain_histogram(data: List[Image], sample_size: int, title: str = None) -> Figure:
    """Compute the pixel value histogram for a collection of images using a sample of the specified size. 

    Args:
        data (List[Image]): List of images
        sample_size (int): Number of image from which the histogram is computed
        title (str, optional): Title shown in the histogram. Defaults to None.

    Returns:
        Figure: _description_
    """
    if sample_size > len(data):
        sample_size = len(data)

    sample = [img.image.flatten() for img in random.sample(data, sample_size)]
    voxels = np.concatenate(sample)

    fig, ax = plt.subplots(1, 1)
    freq, bins, _ = ax.hist(voxels, bins=100, density=True)  # Make histogram
    ax.set_xlabel("Voxel value")
    ax.set_ylabel("Frequency")
    ax.set_title(title)

    return fig
