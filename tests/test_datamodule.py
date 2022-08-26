import torch
import pytest
from torch.utils.data import DataLoader


def test_datamodule_init():
    """Test the initialization of the dataloaders.
    """
    from data.datamodule.datamodule import ContoursDataModule

    dataset_path = "data/contours/expressive"
    dm = ContoursDataModule(dataset_path, 32, 1024, False)

    dm.setup()

    assert isinstance(dm.train_dataloader(), DataLoader)
    assert isinstance(dm.val_dataloader(), DataLoader)
    assert isinstance(dm.test_dataloader(), DataLoader)


def test_datamodule_element_shape():
    """Test the shape of the elements of datamodule.
    """
    from data.datamodule.datamodule import ContoursDataModule

    dataset_path = "data/contours/expressive"
    sample_length = 1024
    batch_size = 32
    dm = ContoursDataModule(dataset_path, batch_size, sample_length, False)

    dm.setup()
    train = dm.train_dataloader()
    assert isinstance(train, DataLoader)

    item = next(iter(train))
    assert item.shape == (batch_size, 2, sample_length)