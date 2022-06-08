import abc
from typing import Union, Tuple, List

import numpy as np
from scipy.ndimage import zoom
import torch


class Transform(abc.ABC):

    def __init__(self) -> None:
        """The method set up the configuration of the transform when is instanciated
        """
        pass

    @abc.abstractmethod
    def __call__(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Apply the transformation to the input data and return it.

        Args:
            data (Union[torch.Tensor, np.ndarray]): input data to transform
        Returns:
            Union[torch.Tensor, np.ndarray]: transformed data
        """
        pass


class Resize(Transform):

    def __init__(self, new_size: Tuple[int, ...], interpolation_order: int = 2) -> None:

        super().__init__()
        self.new_size = new_size
        self.interpolation_order = interpolation_order

    def __call__(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:

        new_shape = np.array(self.new_shape)
        current_shape = np.array(data.shape)
        resize_factor = new_shape / current_shape

        new_data = zoom(data, resize_factor, order=self.interpolation_order)

        return new_data


class Identity(Transform):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        return data


class Compose(Transform):

    def __init__(self, transforms: List[Transform]) -> None:

        super().__init__()

        if transforms is None or len(transforms) == 0:
            transforms = [Identity]

        self.transforms = transforms

    def __call__(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:

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

    def __call__(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Add a noise filter to the data.

        Args:
            data (Union[torch.Tensor, np.ndarray])

        Returns:
            Union[torch.Tensor, np.ndarray]
        """
        noise = np.random.normal(loc=self.mean, scale=self.sd, size=data.shape)
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

    def __call__(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:

        crop_range = self.crop_range
        new_range = self.new_range

        if crop_range is None:
            crop_range = (data.min(), data.max())
        if new_range is None:
            new_range = (-1, 1)

        data[data < crop_range[0]] = crop_range[0]
        data[data > crop_range[1]] = crop_range[1]

        return (data - crop_range[0]) / (crop_range[1] - crop_range[0]) * (new_range[1] - new_range[0]) + new_range[0]
