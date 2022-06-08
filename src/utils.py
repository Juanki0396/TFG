
import datetime
import os
import random
import time
from typing import Callable, Any, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from .data.image import Image


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


def load_xray_data(dataset_path: str = "Data/x_ray") -> Tuple(List[Image], List[Image]):
    """Load Chest x rays train and test datasets.

    Args:
        dataset_path (str, optional): Defaults to "Data/x_ray".

    Returns:
        Tuple(List[Image], List[Image]): Training data y test data
    """
    train_path = os.path.join(dataset_path, "train_SERAM.npy")
    test_path = os.path.join(dataset_path, "test1_SERAM.npy")

    datasets = []

    for path in [train_path, test_path]:
        array = np.load(path, allow_pickle=True)
        data = [Image(img, label) for img, label in array]
        datasets.append(data)

    train_data, test_data = datasets

    return train_data, test_data


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


def obtain_histogram(data: List[Image], sample_size: int, title: str = None, percentage: float = 0.99) -> Figure:
    """Compute the pixel value histogram for a collection of images using a sample of the specified size. Also computes
    the set in which a certain percentage of the total pixel resides.

    Args:
        data (List[Image]): List of images
        sample_size (int): Number of image from which the histogram is computed
        title (str, optional): Title shown in the histogram. Defaults to None.
        percentage (float, optional): Defaults to 0.99.

    Returns:
        Figure: _description_
    """

    sample = [img.image.flatten() for img in random.sample(data, sample_size)]
    voxels = np.concatenate(sample)

    fig, ax = plt.subplots(1, 1)
    freq, bins, _ = ax.hist(voxels, bins=100, density=True)  # Make histogram
    area = hist_integral(freq, bins)  # Integral for each x
    most_dist = bins[len(area[area < percentage])]  # Take the max pixel value that fits in the percentage of the total
    ax.set_xlabel("Voxel value")
    ax.set_ylabel("Frequency")
    ax.set_title(title)

    print(f"{title} -> {percentage * 100:.1f}% of voxels are between [{bins[0]:.1f}, {most_dist:.1f}]")

    return fig
