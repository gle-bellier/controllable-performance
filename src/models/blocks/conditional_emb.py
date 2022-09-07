from random import sample
import torch
import torch.nn as nn
from models.blocks.conv_block import ConvBlock
from models.blocks.gamma_beta import GammaBeta


class ConditionEmbedder(nn.Module):

    def __init__(self, sample_length: int, input_length: int, in_c: int,
                 conditional: bool) -> None:
        """Initialize ConditionalEmbs. Block for 
        conditional embedding.

        Args:
            sample_length (int): length L of the samples of shape (B C L).
            in_c (int): number of channels in the input tensor.
            conditional (bool): if set to True then conditional contours are taken 
            into account.
        """
        super(ConditionEmbedder, self).__init__()
        self.noise_embedder = GammaBeta(512, in_c)
        self.in_conv = ConvBlock(input_length, in_c, in_c)
        self.out_conv = ConvBlock(input_length, in_c, in_c)

        self.conditional = conditional
        if self.conditional:
            self.contours_embedder = nn.Sequential(
                ConvBlock(sample_length, 2, in_c),
                nn.Linear(sample_length, input_length), nn.SiLU())

    def forward(self, x: torch.Tensor, contours: torch.Tensor,
                noise_scale: torch.Tensor) -> torch.Tensor:
        """Compute pass forward.

        Args:
            x (torch.Tensor): input tensor of shape (B C L).
            contours (torch.Tensor): conditioning contours of shape 
            (B 2 sample_length).
            noise_scale (torch.Tensor): noise scale embedding of
            shape (B N).

        Returns:
            torch.Tensor: output tensor of shape (B C L). 
        """
        if self.conditional:
            x = x - self.contours_embedder(contours)

        res = self.in_conv(x)
        gamma, beta = self.noise_embedder(noise_scale)
        res = gamma * res + beta

        return self.out_conv(res) + x
