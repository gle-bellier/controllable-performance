import torch
import torch.nn as nn

from models.blocks.conv_block import ConvBlock


class ResNetBlock(nn.Module):

    def __init__(self, input_length: int, in_c: int) -> None:
        """Initialize ResNetBlock.

        Args:
            input_length (int): length L of the sample of shape (B C L).
            in_c (int): number of input channels.
        """
        super(ResNetBlock, self).__init__()
        self.main = nn.Sequential(ConvBlock(input_length, in_c, in_c),
                                  ConvBlock(input_length, in_c, in_c))
        self.res = nn.Sequential(
            nn.Conv1d(in_channels=in_c,
                      out_channels=in_c,
                      kernel_size=1,
                      padding=0), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pass forward.

        Args:
            x (torch.Tensor): input tensor of shape (B C L).

        Returns:
            torch.Tensor: ouput tensor of shape (B C L).
        """
        return self.main(x) + self.res(x)