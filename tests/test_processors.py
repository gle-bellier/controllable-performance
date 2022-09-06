import pytest
import torch

from data.datamodule.datamodule import ContoursDataModule
from data.dataprocessor.minmax_processor import MinMaxProcessor


def test_minmax_processor_init():

    train_path = "data/contours/expressive/train.pickle"
    ddsp_path = "ddsp_violin.ts"
    min_max = MinMaxProcessor(data_range=(-1, 1),
                              train_path=train_path,
                              ddsp_path=ddsp_path)

    assert isinstance(min_max, MinMaxProcessor)


@pytest.mark.parametrize("data_range", [(-1, 1), (0, 1), (-1, 0)])
def test_minmax_transform(data_range):

    # init min max scaler
    train_path = "data/contours/expressive/train.pickle"
    ddsp_path = "ddsp_violin.ts"
    min_max = MinMaxProcessor(data_range=data_range,
                              train_path=train_path,
                              ddsp_path=ddsp_path)

    # init train dataset
    dataset_path = "data/contours/expressive"
    sample_length = 1024
    batch_size = 32
    dm = ContoursDataModule(dataset_path, batch_size, sample_length, False)
    dm.setup()
    train = dm.train_dataloader()

    for i, sample in enumerate(iter(train), 0):
        sample = min_max.transform(sample)
        assert sample.min() >= data_range[0]
        assert sample.max() <= data_range[1]


@pytest.mark.parametrize("data_range", [(-1, 1), (0, 1), (-1, 0)])
def test_minmax_inverse_transform(data_range):

    # init min max scaler
    train_path = "data/contours/expressive/train.pickle"
    ddsp_path = "ddsp_violin.ts"
    min_max = MinMaxProcessor(data_range=data_range,
                              train_path=train_path,
                              ddsp_path=ddsp_path)

    # init train dataset
    dataset_path = "data/contours/expressive"
    sample_length = 1024
    batch_size = 32
    dm = ContoursDataModule(dataset_path, batch_size, sample_length, False)
    dm.setup()
    train = dm.train_dataloader()

    for i, sample in enumerate(iter(train), 0):
        scaled = min_max.transform(sample)
        rec = min_max.inverse_transform(scaled)

        assert torch.max(sample - rec) < 1e-5