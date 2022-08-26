import torch
import torch.nn as nn

from models.blocks.conv_block import ConvBlock


class ResNetBlock(nn.Module):

    def __init__(self, channels: int, sample_length: int) -> None:
        """Initialize ResNetBlock.

        Args:
            channels (int): number of channels in the convolutions.
            sample_length (int): length L of the sample of shape (B C L).
        """
        super(ResNetBlock, self).__init__()
        self.main = nn.Sequential(ConvBlock(channels, sample_length),
                                  ConvBlock(channels, sample_length))
        self.res = nn.Conv1d(in_channels=channels,
                             out_channels=channels,
                             kernel_size=1,
                             padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pass forward.

        Args:
            x (torch.Tensor): input tensor of shape (B C L).

        Returns:
            torch.Tensor: ouput tensor of shape (B C L).
        """
        return self.main(x) + self.res(x)