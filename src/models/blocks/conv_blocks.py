import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, channels: int, n_sample: int) -> None:
        """Initialize ConvBlock.

        Args:
            channels (int): number of channels in the convolution.
            n_sample (int): length L of the sample of shape (B C L).
        """
        super(ConvBlock, self).__init__()
        #self.ln = nn.LayerNorm([channels, n_sample])

        self.gn = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.conv = nn.Conv1d(in_channels=channels,
                              out_channels=channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.swish = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pass forward.

        Args:
            x (torch.Tensor): input tensor of shape (B C L).

        Returns:
            torch.Tensor: output tensor of shape (B C L).
        """
        x = self.gn(x)
        x = self.swish(x)
        return self.conv(x)
