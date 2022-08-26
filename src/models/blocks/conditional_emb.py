import torch
import torch.nn as nn
from .conv_blocks import ConvBlock
from .gamma_beta import GammaBeta


class ConditionalEmbs(nn.Module):

    def __init__(self, n_sample: int, in_c: int) -> None:
        """Initialize ConditionalEmbs. Block for 
        conditional embedding.

        Args:
            in_c (int): number of channels in the input tensor.
        """
        super(ConditionalEmbs, self).__init__()
        self.emb = GammaBeta(512, in_c)
        self.in_conv = ConvBlock(in_c, n_sample)
        self.out_conv = ConvBlock(in_c, n_sample)

    def forward(self, x: torch.Tensor,
                noise_scale: torch.Tensor) -> torch.Tensor:
        """Compute pass forward.

        Args:
            x (torch.Tensor): input tensor of shape (B C L).
            noise_scale (torch.Tensor): noise scale embedding of
            shape (B N).

        Returns:
            torch.Tensor: output tensor of shape (B C L). 
        """

        res = self.in_conv(x)
        gamma, beta = self.emb(noise_scale)
        res = gamma * res + beta

        return self.out_conv(res) + x