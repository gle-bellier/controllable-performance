import numpy as np
import pytest

from typing import Tuple, List


@pytest.mark.parametrize("contours_length, sample_length, expected",
                         [(10240, 1024, (2, 1024)), (3494, 512, (2, 512)),
                          ((2024), 256, (2, 256))])
def test_dataset_item_shape(contours_length: int, sample_length: int,
                            expected: Tuple[int]) -> None:

    from data.dataset.dataset import ContoursDataset

    f0 = np.random.random(contours_length)
    lo = np.random.random(contours_length)
    dataset = ContoursDataset(f0, lo, sample_length, "train")

    for i in range(len(dataset)):
        assert dataset[i].shape == expected
