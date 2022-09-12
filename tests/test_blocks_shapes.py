import torch
import pytest
from typing import Tuple


@pytest.mark.parametrize(
    "input_shape, condition_shape, sample_length, input_length, in_c",
    [((13, 2, 2048), (13, 2, 1024), 1024, 2048, 2),
     ((13, 4, 512), (13, 2, 1024), 1024, 512, 4),
     ((13, 32, 4096), (13, 2, 1024), 1024, 4096, 32)])
def test_noise_embedder_shape(input_shape: Tuple[int],
                              condition_shape: Tuple[int], sample_length: int,
                              input_length: int, in_c: int) -> None:
    """Test shape of output of conditional embedding block.

    Args:
        input_shape (Tuple[int]): input shape (B C_in L_in).
        condition_shape (Tuple[int]): contours shape (B C L).
        sample_length (int): length L of the sample tensor.
        input_length (int): length L of the input tensor.
        in_c (int): number of channels used for convolutions.
    """

    from models.blocks.conditional_emb import ConditionEmbedder

    x = torch.randn(input_shape)
    condition = torch.randn(condition_shape)
    noise_scale = torch.randn(input_shape[0], 512)

    emb = ConditionEmbedder(sample_length,
                            input_length,
                            in_c,
                            conditional=False,
                            activation=torch.nn.LeakyReLU)

    assert emb(x, condition, noise_scale).shape == input_shape


@pytest.mark.parametrize(
    "input_shape, condition_shape, sample_length, input_length, in_c",
    [((13, 2, 2048), (13, 2, 1024), 1024, 2048, 2),
     ((13, 4, 512), (13, 2, 1024), 1024, 512, 4),
     ((13, 32, 4096), (13, 2, 1024), 1024, 4096, 32)])
def test_condition_embedder_shape(input_shape: Tuple[int],
                                  condition_shape: Tuple[int],
                                  sample_length: int, input_length: int,
                                  in_c: int) -> None:
    """Test shape of output of conditional embedding block.

    Args:
        input_shape (Tuple[int]): input shape (B C_in L_in).
        condition_shape (Tuple[int]): contours shape (B C L).
        sample_length (int): length L of the sample tensor.
        input_length (int): length L of the input tensor.
        in_c (int): number of channels used for convolutions.
    """

    from models.blocks.conditional_emb import ConditionEmbedder

    x = torch.randn(input_shape)
    condition = torch.randn(condition_shape)
    noise_scale = torch.randn(input_shape[0], 512)

    emb = ConditionEmbedder(sample_length,
                            input_length,
                            in_c,
                            conditional=True,
                            activation=torch.nn.LeakyReLU)

    assert emb(x, condition, noise_scale).shape == input_shape


@pytest.mark.parametrize("input_shape, input_length, in_c, out_c, expected",
                         [((13, 2, 2048), 2048, 2, 4, (13, 4, 2048)),
                          ((13, 4, 512), 512, 4, 32, (13, 32, 512)),
                          ((13, 32, 4096), 4096, 32, 2, (13, 2, 4096))])
def test_conv_block_shape(input_shape: Tuple[int], input_length: int,
                          in_c: int, out_c: int, expected: Tuple[int]) -> None:
    """Test shape of output of convolutional block.

    Args:
        input_shape (Tuple[int]): input shape (B C_in L_in).
        input_length (int): length L_in of the input tensor.
        in_c (int): number of input channels used for convolutions.
        out_c (int): number of output channels used for convolutions.
        expected (Tuple[int]): expected output shape (B C_out L_in)
    """

    from models.blocks.conv_block import ConvBlock

    x = torch.randn(input_shape)

    conv = ConvBlock(input_length, in_c, out_c, torch.nn.LeakyReLU)

    assert conv(x).shape == expected


@pytest.mark.parametrize("input_shape, input_length, in_c, expected",
                         [((13, 2, 2048), 2048, 2, (13, 2, 2048)),
                          ((13, 4, 512), 512, 4, (13, 4, 512)),
                          ((13, 32, 4096), 4096, 32, (13, 32, 4096))])
def test_resnet_block_shape(input_shape: Tuple[int], input_length: int,
                            in_c: int, expected: Tuple[int]) -> None:
    """Test shape of output of convolutional block.

    Args:
        input_shape (Tuple[int]): input shape (B C_in L_in).
        input_length (int): length L_in of the input tensor.
        in_c (int): number of input channels used for convolutions.
        out_c (int): number of output channels used for convolutions.
        expected (Tuple[int]): expected output shape (B C_out L_in)
    """

    from models.blocks.resnet_block import ResNetBlock

    x = torch.randn(input_shape)

    conv = ResNetBlock(input_length, in_c, torch.nn.LeakyReLU)

    assert conv(x).shape == expected