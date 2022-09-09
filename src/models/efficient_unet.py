import numpy as np
from typing import List
import torch
import torch.nn as nn

from models.blocks import RFFBlock, UpBlock, DownBlock


class EfficientUnet(nn.Module):

    def __init__(self, sample_length: int, channels: List[int],
                 strides: List[int], num_resnets: List[int],
                 conditional: bool) -> None:
        """Initialize EfficientUnet. Architecture inspired by the efficient unet
        used in Imagen article. Does not include attention layers.

        Args:
            sample_length (int): length of the samples.
            channels (List[int]): list of channels of the convolution blocks.
            strides (List[int]): list of the strides of the convolution blocks.
            num_resnets (List[int]): list of the number of resnets in each of the 
            down/upsampling block (symetrie).
            conditional (bool): if set to True then conditioning signal is taken into
            account in addition of the noise scale embedding.
        """

        super(EfficientUnet, self).__init__()

        h_dims = [32, 128, 512]
        input_lengths = [sample_length] + [
            sample_length // cs for cs in np.cumprod(strides)
        ]
        self.noise_rff = RFFBlock(input_dim=1,
                                  rff_dim=32,
                                  mlp_hidden_dims=h_dims)

        self.dwn_blocks = nn.ModuleList([
            DownBlock(sample_length=sample_length,
                      input_length=input_length,
                      in_c=in_c,
                      out_c=out_c,
                      stride=stride,
                      num_resnets=n,
                      conditional=conditional)
            for input_length, in_c, out_c, stride, n in zip(
                input_lengths, channels[:-1], channels[1:], strides,
                num_resnets)
        ])
        skip_co = [True] * (len(strides) - 1) + [False]
        self.up_blocks = nn.ModuleList([
            UpBlock(sample_length=sample_length,
                    input_length=input_length,
                    in_c=in_c,
                    out_c=out_c,
                    stride=stride,
                    num_resnets=n,
                    conditional=conditional,
                    skip_co=sc)
            for input_length, in_c, out_c, stride, n, sc in zip(
                input_lengths[::-1], channels[-1:0:-1], channels[-2::-1],
                strides[::-1], num_resnets[::-1], skip_co)
        ])

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

        noise_scale = self.noise_rff(noise_scale)

        dwn_outs = []
        # downsampling
        for dwn in self.dwn_blocks:
            x = dwn(x, condition, noise_scale)
            dwn_outs += [x]

        # upsampling
        for up, skip in zip(self.up_blocks, dwn_outs[::-1]):
            x = up(x, condition, noise_scale, skip)

        return x
