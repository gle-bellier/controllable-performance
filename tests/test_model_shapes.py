import torch
import pytest
from typing import List, Tuple


@pytest.mark.parametrize(
    "input_shape, channels, strides, num_resnets",
    [((13, 2, 1024), [2, 4, 8, 16], [2, 2, 4], [1, 1, 3]),
     ((13, 2, 1024), [2, 4, 8, 8, 16], [2, 2, 4, 8], [1, 1, 3, 5])])
def test_efficient_unet_shape(input_shape: Tuple[int], channels: List[int],
                              strides: List[int],
                              num_resnets: List[int]) -> None:
    """Test shape of output of the model.

    Args:
        input_shape (Tuple[int]): input shape (B C_in L_in).
        stride (int): stride of the up block.
        expected (Tuple[int]): expected output shape (B C_out L_out).
    """

    from models.efficient_unet import EfficientUnet

    batch_size = input_shape[0]
    sample_length = input_shape[-1]

    unet = EfficientUnet(sample_length=sample_length,
                         channels=channels,
                         strides=strides,
                         num_resnets=num_resnets,
                         conditional=True)

    x = condition = torch.randn(input_shape)
    noise_scale = torch.randn(batch_size, 1)

    assert unet(x, condition, noise_scale).shape == input_shape


@pytest.mark.parametrize(
    "input_shape, channels, strides, num_resnets",
    [((13, 2, 1024), [2, 4, 8, 16], [2, 2, 4], [1, 1, 3]),
     ((13, 2, 1024), [2, 4, 8, 8, 16], [2, 2, 4, 8], [1, 1, 3, 5])])
def test_uncond_efficient_unet_shape(input_shape: Tuple[int],
                                     channels: List[int], strides: List[int],
                                     num_resnets: List[int]) -> None:
    """Test shape of output of the model.

    Args:
        input_shape (Tuple[int]): input shape (B C_in L_in).
        stride (int): stride of the up block.
        expected (Tuple[int]): expected output shape (B C_out L_out).
    """

    from models.efficient_unet import EfficientUnet

    batch_size = input_shape[0]
    sample_length = input_shape[-1]

    unet = EfficientUnet(sample_length=sample_length,
                         channels=channels,
                         strides=strides,
                         num_resnets=num_resnets,
                         conditional=False)

    x = condition = torch.randn(input_shape)
    noise_scale = torch.randn(batch_size, 1)

    assert unet(x, condition, noise_scale).shape == input_shape