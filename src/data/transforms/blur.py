import torch
import math
import numpy as np
import scipy.linalg as lin

from data.transforms.transform import ConditionTransform


class Blur(ConditionTransform):
    """Conditioning contours blurring method.
    """

    def __init__(self, sample_length=1024, sig_f0=4., sig_lo=8.) -> None:
        """Initialize ConditionTransform.
        """
        super().__init__()
        self.M_f0 = self.blur_matrix(sample_length, sig_f0)
        self.M_lo = self.blur_matrix(sample_length, sig_lo)

    def gaussian(self, x, mu, sigma):

        y = np.exp(-((x - mu)**2) / (2 * sigma**2))
        return y / np.sqrt(2 * math.pi * sigma**2)

    def blur_matrix(self, sample_length, sigma):
        time = np.arange(0, sample_length)
        kernel = self.gaussian(time, sample_length // 2, sigma)
        M = lin.toeplitz(kernel)
        M = np.roll(M, sample_length // 2, axis=0)
        return torch.from_numpy(M)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transform to a input tensor.

        Args:
            x (torch.Tensor): conditioning tensor of shape
            (B C L).
            
        Returns:
            torch.Tensor: modified condition.
        """
        f0, lo = x.split(1, -2)
        M_f0 = self.M_f0.type_as(x)
        M_lo = self.M_lo.type_as(x)

        # apply blurring
        f0 = f0 @ M_f0.T
        lo = lo @ M_lo.T

        out = torch.cat((f0, lo), dim=-2)

        return out
