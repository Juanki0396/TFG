import abc
from typing import Union

import numpy as np
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

    def __init__(self, max_pixel_value: float = 255, min_pixel_value: float = 0) -> None:
        """Instanciate a DynamicRangeScaling that will change the pixel value window to (0,1) range.


        Args:
            max_pixel_value (float, optional): Defaults to 255.
            min_pixel_value (float, optional): Defaults to 0.
        """
        super().__init__()

        self.max_value = max_pixel_value
        self.min_value = min_pixel_value

    def __call__(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:

        max_ = self.max_value
        min_ = self.max_value

        return (data - min_) / (max_ - min_)
