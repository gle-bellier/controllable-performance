import torch
import torch.nn as nn
from typing import Callable

from models.blocks.conv_block import ConvBlock


class Bottleneck(nn.Module):

    def __init__(self, sample_length: int, input_length: int, in_c: int,
                 hidden_c: int, conditional: bool,
                 activation: Callable) -> None:
        """Inititialize conditional bottleneck for UNet architectures.

        Args:
            sample_length (int): length of the sample.
            input_length (int): length L of the input of shape (B C L).
            in_c (int): number C of input channels.
            hidden_c (int): number of channels of the condition projection.
            conditional (bool): if set to True then the bottleneck is conditioned
            by the contours.
            activation (Callable): activation function.
        """

        super().__init__()

        self.conditional = conditional
        if conditional:
            self.condition_conv = ConvBlock(input_length=sample_length,
                                            in_c=2,
                                            out_c=hidden_c,
                                            activation=activation)

            self.condition_lin = nn.Sequential(
                nn.Linear(sample_length, input_length), activation())

            self.out_conv = ConvBlock(input_length=input_length,
                                      in_c=in_c + hidden_c,
                                      out_c=in_c,
                                      activation=activation)
        else:
            self.conv = ConvBlock(input_length=input_length,
                                  in_c=in_c,
                                  out_c=in_c,
                                  activation=activation)

    def forward(self, x: torch.Tensor, contours: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.

        Args:
            x (torch.Tensor): input tensor of shape (B C L).
            contours (torch.Tensor): condition contours of shape 
            (B 2 sample_length).

        Returns:
            torch.Tensor: output tensor of shape (B C L).
        """

        if self.conditional:

            contours = self.condition_conv(contours)
            contours = self.condition_lin(contours)

            out = torch.cat([contours, x], -2)
            out = self.out_conv(out)

        else:
            out = self.conv(x)

        return out
