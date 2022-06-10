from __future__ import annotations
from PIL import Image as pil_Image
import os
from typing import Tuple, Union, Iterable, List

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import torch
from scipy.ndimage import zoom


class Image:
    """Class that handles black and white images
    """

    # TODO Create copy method

    @classmethod
    def from_path(cls, path: str) -> Image:
        """Creates a Image instance from path to file.

        Args:
            paths (str): path of the image to load.

        Returns:
            Image
        """
        image = np.array(pil_Image.open(path))
        image = cls(image)
        return image

    @staticmethod
    def create_image_grid(images: List[Image], grid_size: Tuple[int, int]) -> Figure:

        fig = plt.figure(figsize=grid_size)
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(1, len(images)),  # creates 1xn grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
                         )

        for ax, im in zip(grid, images):
            # Iterating over the grid returns the Axes.
            ax.imshow(im.image)
            ax.axis("off")
            ax.set_title(im.label)

        return fig

    @staticmethod
    def standarize_bw_image(image: np.ndarray) -> np.ndarray:
        """Convert an image to black and white with 2 dims.

        Args:
            image (np.ndarray): image to convert

        Returns:
            np.ndarray: bw image
        """

        if image.ndim not in (2, 3):
            raise ValueError(f"Image has {image.ndim} dimensions. Images should have 2 or 3 dimensions.")

        if image.ndim == 3:

            channel_axis = np.argmin(image.shape)  # Channel should be the smallest dim for 2d images
            image = image.mean(axis=channel_axis)

        return image

    def __init__(self, image: Union[np.ndarray, torch.Tensor], label: Union[str, float, int] = None) -> None:

        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        self.image = Image.standarize_bw_image(image)
        self.label = label

    @property
    def torch_tensor(self) -> torch.Tensor:
        """Return the bw image as a torch tensor with a channel dimension. This allow the tensor to be
        used by CNNs.

        Returns:
            torch.Tensor
        """
        numpy_copy = self.image.copy()
        tensor = torch.from_numpy(numpy_copy).unsqueeze(dim=0).repeat([3, 1, 1])  # dims (1, H, W)

        return tensor

    def reshape(self, new_shape: Tuple[int, int], interpolation_order: int = 3) -> Image:
        """Reshape the image inplace to the selected shape.

        Args:
            new_shape (Tuple[int,int]): (H,W)
            interpolation_order (int, optional):  Defaults to 3.
        """

        if interpolation_order not in (0, 1, 2, 3, 4, 5):
            raise ValueError(f"Interpolation order must be an integer between 0 and 5.")

        new_shape = np.array(new_shape)
        current_shape = np.array(self.image.shape)
        resize_factor = new_shape / current_shape

        self.image = zoom(self.image, resize_factor, order=interpolation_order)

        return self

    def save_image(self, dir_path: str, file_name: str) -> None:
        """Save the image in the specified path. File name can contain the extension, if not it is set to png.

        Args:
            dir_path (str)
            file_name (str)

        """

        if not os.path.exists(dir_path):
            raise OSError(f"Path {dir_path} does not exists")

        if "." not in file_name:
            file_name += ".png"

        path = os.path.join(dir_path, file_name)

        image = pil_Image.fromarray(self.image).convert("I")
        image.save(path)

    def plot(self, title: str = None, cmap: str = "gray", figsize: Tuple[int, int] = (14, 8)) -> Figure:
        """Plot the image with the selected tittle, colormap and figsize.

        Args:
            tittle (str, optional): Defaults to None.
            cmap (str, optional): Defaults to "gray".
            figsize (Tuple[int,int], optional): Defaults to (14,8).
        """

        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=figsize)
        axes.imshow(self.image, cmap=cmap)
        axes.axis("off")
        if title is None:
            title = self.label
        axes.set_title(title)

        return fig
