import torch
import torch.nn as nn

from models.blocks.conditional_emb import ConditionEmbedder
from models.blocks.resnet_block import ResNetBlock
import numpy as np


class UpBlock(nn.Module):

    def __init__(self,
                 sample_length: int,
                 input_length: int,
                 in_c: int,
                 out_c: int,
                 stride: int,
                 num_resnets: int,
                 conditional: bool,
                 skip_co=True) -> None:
        """Initialize UpBlock.

        Args:
            sample_length (int): length L' of the sample of shape (B 2 L').
            input_length (int): length L of the input of shape (B C L).
            in_c (int): number of input channels in the convolution.
            out_c (int): number of output channels in the convolution.
            stride (int): stride of the convolution.
            num_resnets (int): number of resnet in the upsampling block.
            conditional (bool): if set to True then the conditoning signal is 
            taken into account in the ConditionEmbedder in addition to the 
            noise_scale.
            skip_co (bool): if block takes skip connection ouput as input.
            Default to True.
        """
        super(UpBlock, self).__init__()
        self.skip_co = skip_co
        self.num_resnets = num_resnets
        self.embedder = ConditionEmbedder(sample_length=sample_length,
                                          input_length=input_length,
                                          in_c=in_c,
                                          conditional=conditional)
        self.residual = nn.Sequential(*[
            ResNetBlock(input_length=input_length, in_c=in_c)
            for _ in range(num_resnets)
        ])
        self.up = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_c,
                               out_channels=out_c,
                               kernel_size=3,
                               stride=stride,
                               padding=self.get_padding(3, stride, 1),
                               output_padding=stride - 1), nn.SiLU())

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

    def forward(self, x: torch.Tensor, condition: torch.Tensor,
                noise_scale: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Compute pass forward.

        Args:
            x (torch.Tensor): input tensor of shape (B C L).
            condition (torch.Tensor): condition tensor of shape (B 2 sample_length).
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
        x = self.embedder(x, condition, noise_scale)
        x = self.residual(x)
        return self.up(x)