import torch
import torch.nn as nn

from models.blocks.conditional_emb import ConditionalEmbs
from models.blocks.resnet_block import ResNetBlock
import numpy as np


class UpBlock(nn.Module):

    def __init__(self,
                 n_sample: int,
                 in_c: int,
                 out_c: int,
                 stride: int,
                 num_resnets: int,
                 skip_co=True) -> None:
        """Initialize UpBlock.

        Args:
            in_c (int): number of input channels in the convolution.
            out_c (int): number of output channels in the convolution.
            stride (int): stride of the convolution.
            num_resnets (int): number of resnet in the upsampling block.
            skip_co (bool): if block takes skip connection ouput as input.
            Default to True.
        """
        super(UpBlock, self).__init__()
        self.skip_co = skip_co
        self.num_resnets = num_resnets
        self.noise_emb = ConditionalEmbs(n_sample=n_sample, in_c=in_c)
        self.residual = nn.Sequential(
            *[ResNetBlock(in_c, n_sample) for _ in range(num_resnets)])
        self.up = nn.ConvTranspose1d(in_channels=in_c,
                                     out_channels=out_c,
                                     kernel_size=3,
                                     stride=stride,
                                     padding=self.get_padding(3, stride, 1),
                                     output_padding=stride - 1)

    def get_padding(self, kernel_size: int, stride: int, dilation: int) -> int:
        """Return size of the padding needed.
        Args:
            kernel_size (int): kernel size of the convolutional layer
            stride (int): stride of the convolutional layer
            dilation (int): dilation of the convolutional layer
        Returns:
            int: padding
        """

        full_kernel = (kernel_size - 1) * dilation + 1
        return full_kernel // 2

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                noise_scale: torch.Tensor) -> torch.Tensor:
        """Compute pass forward.

        Args:
            x (torch.Tensor): input tensor of shape (B C L).
            skip (torch.Tensor): input tensor from skip connection
            of shape (B C L).
            noise_scale (torch.Tensor): noise scale embedding of shape
            (B N=512).

        Returns:
            torch.Tensor: output tensor of shape (B C' L').
        """
        if self.skip_co:
            if self.num_resnets > 3:
                skip = skip / np.sqrt(2)
            x = x + skip
        x = self.noise_emb(x, noise_scale)
        x = self.residual(x)
        return self.up(x)