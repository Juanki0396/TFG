import abc
from typing import Union, Tuple, List

import numpy as np
from scipy.ndimage import zoom, gaussian_filter
import torch


class Transform(abc.ABC):

    def __init__(self) -> None:
        """The method set up the configuration of the transform when is instanciated
        """
        pass

    @abc.abstractmethod
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the transformation to the input data and return it.

        Args:
            data (torch.Tensor): input data to transform
        Returns:
            torch.Tensor: transformed data
        """
        pass


class Resize(Transform):

    def __init__(self, new_size: Tuple[int, ...], interpolation_order: int = 2) -> None:

        super().__init__()
        self.new_size = new_size
        self.interpolation_order = interpolation_order

    def __call__(self, data: torch.Tensor) -> torch.Tensor:

        new_shape = np.array(self.new_size)
        current_shape = np.array(data.shape)

        if new_shape.shape[0] == current_shape.shape[0] - 1:
            new_shape = np.insert(new_shape, 0, 3)

        resize_factor = new_shape / current_shape

        new_data = zoom(data, resize_factor, order=self.interpolation_order)
        new_data = torch.from_numpy(new_data)

        return new_data


class Identity(Transform):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data


class Compose(Transform):

    def __init__(self, transforms: List[Transform]) -> None:

        super().__init__()

        if transforms is None or len(transforms) == 0:
            transforms = [Identity()]

        self.transforms = transforms

    def __call__(self, data: torch.Tensor) -> torch.Tensor:

        for transform in self.transforms:
            data = transform(data)

        return data


class RandomNoise(Transform):

    def __init__(self, mean: float, sd: float) -> None:
        """Instanciate a RandomNoiseTransform by setting the mean and sd of the normal distribution that
        will generate the noise

        Args:
            mean (float)
            sd (float)
        """
        super().__init__()

        self.mean = mean
        self.sd = sd

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Add a noise filter to the data.

        Args:
            data (Union[torch.Tensor, np.ndarray])

        Returns:
            Union[torch.Tensor, np.ndarray]
        """
        noise = torch.normal(mean=self.mean, std=self.sd, size=data.shape)
        return data + noise


class DynamicRangeScaling(Transform):

    def __init__(self, new_range: Tuple[float, float] = None, crop_range: Tuple[float, float] = None) -> None:
        """Instanciate a DynamicRangeScaling that will change the pixel value. It will crop the original pixel
        range and then rescale to the desired range.


        Args:
            new_range (Tuple[float, float], optional): Output voxel range. Default is (-1,1)
            crop_range (Tuple[float, float], optional): Range to which data is cropped before rescaling.
                                                         Defaults take the min and max values of the image
        """
        super().__init__()

        self.new_range = new_range
        self.crop_range = crop_range

    def __call__(self, data: torch.Tensor) -> torch.Tensor:

        crop_range = self.crop_range
        new_range = self.new_range

        if crop_range is None:
            crop_range = (data.min().item(), data.max().item())
        if new_range is None:
            new_range = (-1, 1)

        data[data < crop_range[0]] = crop_range[0]
        data[data > crop_range[1]] = crop_range[1]

        return (data - crop_range[0]) / (crop_range[1] - crop_range[0]) * (new_range[1] - new_range[0]) + new_range[0]


class Blur(Transform):

    def __init__(self, sd: float) -> None:

        super().__init__()
        self.sd = sd

    def __call__(self, data: torch.Tensor) -> torch.Tensor:

        new_data = gaussian_filter(data, sigma=self.sd)
        new_data = torch.from_numpy(new_data)

        return new_data


class AxisTranspose(Transform):

    def __init__(self, axis: List[int]) -> None:
        super().__init__()
        self.axis = axis

    def __call__(self, data: torch.Tensor) -> torch.Tensor:

        new_data = np.flip(data.numpy(), self.axis)
        new_data = torch.from_numpy(new_data.copy())

        return new_data


class FDATransform(Transform):

    def __init__(self, target_image: torch.Tensor, L: float) -> None:

        super().__init__()
        fft = torch.fft.fft2(target_image)
        self.target_amplitude = fft.abs()
        self.L = L

    def __call__(self, data: torch.Tensor) -> torch.Tensor:

        _, h, w = data.shape
        b = np.floor(np.min((h, w)) * self.L).astype(np.int64)

        src_fft = torch.fft.fft2(data.clone())
        src_amp, src_ang = src_fft.abs(), src_fft.angle()
        src_amp[:, 0:b, 0:b] = self.target_amplitude[:, 0:b, 0:b]
        src_amp[:, 0:b, w-b:w] = self.target_amplitude[:, 0:b, w-b:w]
        src_amp[:, h-b:h, 0:b] = self.target_amplitude[:, h-b:h, 0:b]
        src_amp[:, h-b:h, w-b:w] = self.target_amplitude[:, h-b:h, w-b:w]

        src_fft_mod = src_amp * torch.exp(1j * src_ang)
        new_data = torch.fft.ifft2(src_fft_mod)

        return new_data.real
