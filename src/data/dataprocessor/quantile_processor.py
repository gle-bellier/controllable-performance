import numpy as np
import torch
from typing import List
from einops import rearrange
from sklearn.preprocessing import QuantileTransformer
from data.dataprocessor.dataprocessor import ContoursProcessor


class QuantileProcessor(ContoursProcessor):
    """
    Class used for preprocessing and postprocessing
    and audio generation with DDSP.
    """

    def __init__(self,
                 data_range: List[float],
                 output_distribution: str,
                 train_path: str,
                 ddsp_path: str,
                 sample_length: int,
                 n_quantiles=30) -> None:
        """Initialize quantile transform.

        Args:
            data_range (List[float]): range of processed data.
            ouput_distribution (str): output distribution, either 
            "normal" for gaussian distribution or "uniform" for 
            the uniform distribution. 
            train_path (str): path to the train pickle file.
            ddsp_path (str): path to the pretrained ddsp.
            sample_length (int): length of the input sample. 
            n_quantiles (int, optional): number of quantiles to use 
            for the quantile transform. Defaults to 30.
        """
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.sample_length = sample_length
        # fix the violin pitch range and loudness range
        self.f0_min, self.f0_max = 20, 90
        self.lo_min, self.lo_max = -9, 0
        super().__init__(data_range, train_path, ddsp_path)

    def fit(self):

        f0_scaler = QuantileTransformer(
            n_quantiles=self.n_quantiles,
            output_distribution=self.output_distribution)

        lo_scaler = QuantileTransformer(
            n_quantiles=self.n_quantiles,
            output_distribution=self.output_distribution)
        f0, lo = self._load_train()

        # convert to numpy array
        f0 = f0.detach().cpu().numpy()
        lo = lo.detach().cpu().numpy()

        # clip contours to the violin range
        lo = np.clip(lo, self.lo_min, self.lo_max)
        f0 = np.clip(f0, self.f0_min, self.f0_max)

        # fit scaler
        self.f0_scaler = f0_scaler.fit(f0)
        self.lo_scaler = lo_scaler.fit(lo)

        # fit min and max post transform
        f0 = f0_scaler.transform(f0)
        lo = lo_scaler.transform(lo)

        self.f0_min_post, self.f0_max_post = f0.min(), f0.max()
        self.lo_min_post, self.lo_max_post = lo.min(), lo.max()

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transform to the input tensor. 

        Args:
            x (torch.Tensor): input tensor of shape (B C L).

        Returns:
            torch.Tensor: output tensor of shape (B C L).
        """
        x_np = x.detach().cpu().numpy()
        f0, lo = rearrange(x_np, "b c l -> c (b l) 1")

        f0 = np.clip(f0, self.f0_min, self.f0_max)
        lo = np.clip(lo, self.lo_min, self.lo_max)

        # transform
        f0 = self.f0_scaler.transform(f0)
        lo = self.lo_scaler.transform(lo)

        # rescale to match data range
        scaled_f0 = self._rescale(f0, self.f0_min_post, self.f0_max_post,
                                  self.min, self.max)
        scaled_lo = self._rescale(lo, self.lo_min_post, self.lo_max_post,
                                  self.min, self.max)

        out = rearrange([scaled_f0, scaled_lo],
                        "c (b l) 1 -> b c l",
                        l=self.sample_length)

        return torch.tensor(out).type_as(x)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse transform (i.e. inverse_transform(transform(x)) =x
        to the input tensor.

        Args:
            x (torch.Tensor): input tensor of shape (B C L).

        Returns:
            torch.Tensor: output tensor of shape (B C L).
        """

        x_np = x.detach().cpu().numpy()
        f0, lo = rearrange(x_np, "b c l -> c (b l) 1")

        # rescale f0, lo
        f0 = self._rescale(f0, self.min, self.max, self.f0_min_post,
                           self.f0_max_post)
        lo = self._rescale(lo, self.min, self.max, self.lo_min_post,
                           self.lo_max_post)

        f0 = self.f0_scaler.inverse_transform(f0)
        lo = self.lo_scaler.inverse_transform(lo)

        out = rearrange([f0, lo], " c (b l) 1 -> b c l", l=self.sample_length)

        return torch.tensor(out).type_as(x)
