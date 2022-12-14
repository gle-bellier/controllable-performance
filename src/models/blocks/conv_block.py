import torch
import torch.nn as nn
from typing import Callable


class ConvBlock(nn.Module):

    def __init__(self, input_length: int, in_c: int, out_c: int,
                 activation: Callable) -> None:
        """Initialize ConvBlock.

        Args:
            input_length (int): length L of the input of shape (B C L).
            in_c (int): number of input channels in the convolution.
            out_c (int): number of output channels in the convolution.
            activation (Callable): activation function.
        """
        super(ConvBlock, self).__init__()
        self.input_length = input_length
        self.ln = nn.LayerNorm([out_c, input_length])

        self.conv = nn.Conv1d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        #self.gn = nn.GroupNorm(num_groups=4, num_channels=out_c)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pass forward.

        Args:
            x (torch.Tensor): input tensor of shape (B C L).

        Returns:
            torch.Tensor: output tensor of shape (B C L).
        """
        x = self.conv(x)
        x = self.ln(x)
        return self.activation(x)