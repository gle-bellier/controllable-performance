import pathlib
import numpy as np
from typing import List, Tuple, Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.dataset.dataset import ContoursDataset
from utils.pickle_tools import read_from_pickle


class ContoursDataModule(pl.LightningDataModule):

    def __init__(self,
                 dataset_path: str,
                 batch_size: int,
                 sample_length: int,
                 data_aug=False) -> None:
        super().__init__()

        self.dataset_path = pathlib.Path(dataset_path)
        self.batch_size = batch_size
        self.data_aug = data_aug
        self.sample_length = sample_length

        # datasets will be loaded during setup
        self.train = self.val = self.test = None

    def __load(self, path: pathlib.Path) -> Tuple[np.ndarray]:
        """Load the dataset contained in the pickle data file
        indicated by path.

        Args:
            path (pathlib.Path): path to the pickle data file.

        Returns:
            Tuple[np.ndarray]: tuple of contours of pitch and loudness
        """
        contours = next(read_from_pickle(path))
        return contours["f0"], contours["lo"]

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up and load the different datasets.

        Args:
            stage (Optional[str], optional): extra infos. Defaults to None.
        """

        if self.data_aug:
            train_file = "train_aug.pickle"
        else:
            train_file = "train.pickle"

        train_path = self.dataset_path / pathlib.Path(train_file)
        val_path = self.dataset_path / pathlib.Path("val.pickle")
        test_path = self.dataset_path / pathlib.Path("test.pickle")

        train_f0, train_lo = self.__load(train_path)
        val_f0, val_lo = self.__load(val_path)
        test_f0, test_lo = self.__load(test_path)

        self.train = ContoursDataset(train_f0, train_lo, self.sample_length,
                                     "train")
        self.val = ContoursDataset(val_f0, val_lo, self.sample_length, "train")
        self.test = ContoursDataset(test_f0, test_lo, self.sample_length,
                                    "train")

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        Returns:
            DataLoader: train dataloader.
        """
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=32)

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader.

        Returns:
            DataLoader: validation dataloader.
        """
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=32)

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader.

        Returns:
            DataLoader: test dataloader.
        """
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=32)
