import torch
import torch.nn as nn
from typing import Callable
from models.blocks.conditional_emb import ConditionEmbedder
from models.blocks.resnet_block import ResNetBlock
import numpy as np


class Upsampling(nn.Module):

    def __init__(self, factor: int, in_c: int, out_c: int,
                 activation: Callable) -> None:
        """Initialize upsampling module.

        Args:
            factor (int): upsampling factor.
            in_c (int): input channels.
            out_c (int): output channels.
            activation (Callable): activation function.
        """
        super().__init__()

        self.upsampling = nn.Upsample(scale_factor=factor, mode="linear")
        self.conv = nn.Conv1d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=3,
                              padding=1)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute upsampling.

        Args:
            x (torch.Tensor): input tensor of shape (B C L).

        Returns:
            torch.Tensor: output tensor of shape (B C L*factor).
        """

        x = self.upsampling(x)
        x = self.conv(x)
        return self.activation(x)


class UpBlock(nn.Module):

    def __init__(self,
                 sample_length: int,
                 input_length: int,
                 in_c: int,
                 out_c: int,
                 factor: int,
                 num_resnets: int,
                 activation: Callable,
                 skip_co=True,
                 last=False) -> None:
        """Initialize UpBlock.

        Args:
            sample_length (int): length L' of the sample of shape (B 2 L').
            input_length (int): length L of the input of shape (B C L).
            in_c (int): number of input channels in the convolution.
            out_c (int): number of output channels in the convolution.
            factor (int): interpolation factor.
            num_resnets (int): number of resnet in the upsampling block.
            activation (Callable): activation function.
            taken into account in the ConditionEmbedder in addition to the 
            noise_scale.
            skip_co (bool): if block takes skip connection ouput as input.
            Default to True.
            last (bool): if true then no activation function at the end. 
            Default to False.
        """
        super(UpBlock, self).__init__()
        self.skip_co = skip_co
        self.num_resnets = num_resnets
        self.embedder = ConditionEmbedder(sample_length=sample_length,
                                          input_length=input_length,
                                          in_c=in_c)
        self.residual = nn.Sequential(*[
            ResNetBlock(
                input_length=input_length, in_c=in_c, activation=activation)
            for _ in range(num_resnets)
        ])

        if last:
            activation_last = torch.nn.Identity
        else:
            activation_last = activation

        self.up = Upsampling(factor=factor,
                             in_c=in_c,
                             out_c=out_c,
                             activation=activation_last)

    def forward(self, x: torch.Tensor, noise_scale: torch.Tensor,
                skip: torch.Tensor) -> torch.Tensor:
        """Compute pass forward.

        Args:
            x (torch.Tensor): input tensor of shape (B C L).
            noise_scale (torch.Tensor): noise scale embedding of shape
            (B N=512).
            skip (torch.Tensor): input tensor from skip connection
            of shape (B C L).

        Returns:
            torch.Tensor: output tensor of shape (B C' L').
        """
        if self.skip_co:
            if self.num_resnets > 3:
                skip = skip / np.sqrt(2)
            x = x + skip
        x = self.embedder(x, noise_scale)
        x = self.residual(x)
        return self.up(x)