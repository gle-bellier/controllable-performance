from torch.utils.data import Dataset

import numpy as np
import torch


class ContoursDataset(Dataset):
    """
    Class for expressive fundamental frequency contours. f0 time-series are
    extracted from real humain performances with CREPE at 100hz.
    """

    def __init__(self, f0: np.ndarray, lo: np.ndarray, sample_length: int,
                 mode: str) -> None:
        """Create the dataset object for f0 contours.

        Args:
            f0 (np.ndarray): f0 contours
            lo (np.ndarray): lo contours
            sample_length (int): length of the fundamental frequency contour samples
            mode (str): dataset mode (train/val/eval)
        """

        super().__init__()
        self.f0 = f0
        self.lo = lo
        self.sample_length = sample_length

        assert mode in ['train', 'val',
                        'test'], "mode must be either train, val or eval"
        self.mode = mode
        print(f"{mode} dataset loaded.")

        self.f0, self.lo = f0, lo

    def __len__(self):
        return len(self.f0) // self.sample_length

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Provide a contour sample from the dataset

        Args:
            idx (int): index of the selected sample in the dataset.

        Returns:
            torch.Tensor: contour of shape (C L) where L is 
            sample length and C=2 the number of channels
            (fundamental frequency and loudness).
        """
        idx *= self.sample_length

        # add jitter during training only
        if self.mode == "train":
            idx += np.random.randint(0, self.sample_length // 10)

        # ensure starting idx range even with jitter
        start = max(0, idx)
        start = min(start, len(self.f0) - self.sample_length)

        sample_f0 = torch.Tensor(self.f0[start:start + self.sample_length])
        sample_lo = torch.Tensor(self.lo[start:start + self.sample_length])

        contour = torch.stack([sample_f0, sample_lo])

        return contour