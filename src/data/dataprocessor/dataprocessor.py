import pathlib
import torch
import torch.nn as nn
from typing import Tuple
from einops import rearrange
from utils.pickle_tools import read_from_pickle


class ContoursProcessor():

    def __init__(self, data_range: Tuple[float], train_path: str,
                 ddsp_path: str) -> None:
        super().__init__()
        self.min, self.max = data_range
        self.train_path = pathlib.Path(train_path)

        self.ddsp = self.load_ddsp(ddsp_path)

        # fit scaler at init
        self.fit()

    def load_ddsp(self, ddsp_path: str) -> nn.Module:
        """Load the ddsp module used for sound synthesis.

        Args:
            ddsp_path (str): path to the pretrained ddsp.

        Returns:
            nn.Module: ddsp module.
        """
        if ddsp_path is None:
            return None

        else:
            ddsp = torch.jit.load(ddsp_path).cuda()
            # freeze ddsp
            for p in ddsp.parameters():
                p.requires_grad = False

            return ddsp

    def _load_train(self) -> Tuple[torch.Tensor]:
        """Load the dataset contained in the pickle data file
        indicated by path.

        Args:
            path (pathlib.Path): path to the pickle data file.

        Returns:
            Tuple[torch.Tensor]: tuple of contours of pitch and loudness
            of shape (L, 1).
        """
        contours = next(read_from_pickle(self.train_path))
        f0 = contours["f0"].reshape(-1, 1)
        lo = contours["lo"].reshape(-1, 1)
        return torch.Tensor(f0), torch.Tensor(lo)

    def _rescale(self, x: torch.Tensor, o_m: float, o_M: float, n_m: float,
                 n_M: float) -> torch.Tensor:
        """Scale a tensor to a given range.

        Args:
            x (torch.Tensor): input tensor (of any shape).
            o_m (float): minimum of the input tensor.
            o_M (float): maximum of the input tensor.
            n_m (float): new minimum.
            n_M (float): new maximum.

        Returns:
            torch.Tensor: rescaled tensor (of any shape).
        """
        return (n_M - n_m) / (o_M - o_m) * (x - o_M) + n_M

    def contours_to_wav(self, x: torch.Tensor) -> torch.Tensor:
        """Contours to audio converting

        Args:
            x (torch.Tensor): contour of (f0, lo) of shape 
            (B C L) with C=2.

        Returns:
            torch.Tensor: corresponding waveform of shape (B L*sr)
        """
        # we need loudness contour to generate audio
        f0, lo = rearrange(x, "b c l -> c b l 1", c=2)
        # rescale to the frequency range
        f0 = 440 * torch.pow(2, (f0 - 69) / 12)
        # artificialy add 2db to the loudness contours
        return self.ddsp(f0, lo + 1.5).squeeze(-1)

    def fit(self):
        """Fit the scaler to the training dataset.
        """
        raise NotImplementedError

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transform to the input tensor. 

        Args:
            x (torch.Tensor): input tensor of shape (B C L).

        Raises:
            NotImplementedError: error if not written.

        Returns:
            torch.Tensor: output tensor of shape (B C L).
        """
        raise NotImplementedError

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse transform (i.e. inverse_transform(transform(x)) =x
        to the input tensor.

        Args:
            x (torch.Tensor): input tensor of shape (B C L).

        Raises:
            NotImplementedError: error if not written.

        Returns:
            torch.Tensor: output tensor of shape (B C L).
        """
        raise NotImplementedError
