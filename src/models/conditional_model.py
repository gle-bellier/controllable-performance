import numpy as np
from typing import List
import torch
import torch.nn as nn

from .blocks import RFFBlock, UpBlock, DownBlock


class EfficientUnet(nn.Module):

    def __init__(self, sample_length: int, channels: List[int],
                 strides: List[int], num_resnets: List[int]) -> None:
        """Initialize EfficientUnet. Architecture inspired by the efficient unet
        used in Imagen article. Does not include attention layers.

        Args:
            sample_length (int): length of the samples.
            channels (List[int]): list of channels of the convolution blocks.
            strides (List[int]): list of the strides of the convolution blocks.
            num_resnets (List[int]): list of the number of resnets in each of the 
            down/upsampling block (symetrie).
        """

        super(EfficientUnet, self).__init__()

        h_dims = [32, 128, 512]
        lengths_sample = [sample_length // cs for cs in np.cumprod(strides)]
        self.embedding = RFFBlock(input_dim=1,
                                  rff_dim=32,
                                  mlp_hidden_dims=h_dims)

        self.pre = nn.Sequential(
            nn.Conv1d(channels[0], channels[0], kernel_size=1, padding=0),
            nn.SiLU())
        self.dwn_blocks = nn.ModuleList([
            DownBlock(sample_length, in_c, out_c, stride, num_resnets=n)
            for sample_length, in_c, out_c, stride, n in zip(
                lengths_sample, channels[:-1], channels[1:], strides,
                num_resnets)
        ])
        skip_co = [True] * (len(strides) - 1) + [False]
        self.up_blocks = nn.ModuleList([
            UpBlock(sample_length,
                    in_c,
                    out_c,
                    stride,
                    num_resnets=n,
                    skip_co=sc)
            for sample_length, in_c, out_c, stride, n, sc in zip(
                lengths_sample[::-1], channels[-1:0:-1], channels[-2::-1],
                strides[::-1], num_resnets[::-1], skip_co)
        ])
        self.post = nn.Conv1d(channels[0],
                              channels[0],
                              kernel_size=1,
                              padding=0)

    def forward(self, x: torch.Tensor, condition: torch.Tensor,
                noise_scale: torch.Tensor) -> torch.Tensor:
        """Compute pass forward.

        Args:
            x (torch.Tensor): tensor of shape (B C L) with C=2
            and L=sample_length.
            condition (torch.Tensor): condition of shape (B C L) with C=2
            and L=sample_length.
            noise_scale (torch.Tensor): noise scale of shape (B 1).

        Returns:
            torch.Tensor: output tensor of shape (B C L) with C=2
            and L=sample_length.
        """

        noise_scale = self.embedding(noise_scale)
        x = self.pre(x)
        dwn_outs = []
        # downsampling
        for dwn in self.dwn_blocks:
            x = dwn(x, noise_scale)
            dwn_outs += [x]

        # upsampling
        for up, skip in zip(self.up_blocks, dwn_outs[::-1]):
            x = up(x, skip, noise_scale)

        return self.post(x)
