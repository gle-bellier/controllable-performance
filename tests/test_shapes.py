import torch
import pytest
from typing import Tuple, List


@pytest.mark.parametrize("input_shape, stride, expected",
                         [((13, 2, 1024), 2, (13, 16, 512)),
                          ((13, 2, 1024), 1, (13, 16, 1024)),
                          ((13, 2, 1024), 4, (13, 4, 256))])
def test_down_block_shape(input_shape: Tuple[int], stride: int,
                          expected: Tuple[int]) -> None:
    """Test shape of output of upsampling block.

    Args:
        input_shape (Tuple[int]): input shape (B C_in L_in).
        stride (int): stride of the up block.
        expected (Tuple[int]): expected output shape (B C_out L_out).
    """

    from models.blocks.down_block import DownBlock

    batch_size = input_shape[0]
    in_c, out_c = input_shape[1], expected[1]
    sample_length = input_shape[-1]

    down = DownBlock(sample_length, in_c, out_c, stride, 2)

    x = torch.randn(input_shape)
    noise_scale = torch.randn(batch_size, 512)

    assert down(x, noise_scale).shape == expected


@pytest.mark.parametrize("input_shape, stride, expected",
                         [((13, 2, 1024), 2, (13, 16, 2048)),
                          ((13, 2, 1024), 1, (13, 16, 1024)),
                          ((13, 2, 1024), 4, (13, 4, 4096))])
def test_up_block_shape(input_shape: Tuple[int], stride: int,
                        expected: Tuple[int]) -> None:
    """Test shape of output of upsampling block.

    Args:
        input_shape (Tuple[int]): input shape (B C_in L_in).
        stride (int): stride of the up block.
        expected (Tuple[int]): expected output shape (B C_out L_out).
    """

    from models.blocks.up_block import UpBlock

    batch_size = input_shape[0]
    in_c, out_c = input_shape[1], expected[1]
    sample_length = input_shape[-1]

    up = UpBlock(sample_length, in_c, out_c, stride, 2)

    x = torch.randn(input_shape)
    skip = torch.randn(input_shape)
    noise_scale = torch.randn(batch_size, 512)

    assert up(x, skip, noise_scale).shape == expected


@pytest.mark.parametrize(
    "input_shape, channels, strides, num_resnets",
    [((13, 2, 1024), [2, 4, 8, 16], [2, 2, 4], [1, 1, 3]),
     ((13, 2, 1024), [2, 4, 8, 8, 16], [2, 2, 4, 8], [1, 1, 3, 5])])
def test_efficient_unet_shape(input_shape: Tuple[int], channels: List[int],
                              strides: List[int],
                              num_resnets: List[int]) -> None:
    """Test shape of output of upsampling block.

    Args:
        input_shape (Tuple[int]): input shape (B C_in L_in).
        stride (int): stride of the up block.
        expected (Tuple[int]): expected output shape (B C_out L_out).
    """

    from models.efficient_unet import EfficientUnet

    batch_size = input_shape[0]
    sample_length = input_shape[-1]

    unet = EfficientUnet(sample_length, channels, strides, num_resnets)

    x = condition = torch.randn(input_shape)
    noise_scale = torch.randn(batch_size, 1)

    assert unet(x, condition, noise_scale).shape == input_shape