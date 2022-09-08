import torch

from typing import Tuple, List

from data.dataprocessor.dataprocessor import ContoursProcessor


class MinMaxProcessor(ContoursProcessor):
    """Min Max scaler for contours.

    Args:
        ContoursProcessor () class of scaler for contours.
    """

    def fit(self):
        """Compute the min max values of the train set.
        """
        train_f0, train_lo = self._load_train()

        self.train_f0_min, self.train_f0_max = torch.min(train_f0), torch.max(
            train_f0)
        self.train_lo_min, self.train_lo_max = torch.min(train_lo), torch.max(
            train_lo)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply min max scaling to the input tensor.

        Args:
            x (torch.Tensor): input tensor of shape (B C L).

        Returns:
            torch.Tensor: output tensor of shape (B C L).
        """

        f0 = x[:, 0:1, :]
        lo = x[:, 1:2, :]

        f0 = self._rescale(f0, self.train_f0_min, self.train_f0_max, self.min,
                           self.max)
        lo = self._rescale(lo, self.train_lo_min, self.train_lo_max, self.min,
                           self.max)

        out = torch.cat((f0, lo), dim=-2)
        return out

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse min max scaling to the input tensor.

        Args:
            x (torch.Tensor): input tensor of shape (B C L).

        Returns:
            torch.Tensor: output tensor of shape (B C L).
        """

        f0 = x[:, 0:1, :]
        lo = x[:, 1:2, :]

        f0 = self._rescale(f0, self.min, self.max, self.train_f0_min,
                           self.train_f0_max)
        lo = self._rescale(lo, self.min, self.max, self.train_lo_min,
                           self.train_lo_max)

        out = torch.cat([f0, lo], dim=-2)
        return out
