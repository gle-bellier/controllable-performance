import torch
import torch.nn as nn
from models.blocks.gamma_beta import GammaBeta


class ConditionEmbedder(nn.Module):

    def __init__(self, sample_length: int, input_length: int,
                 in_c: int) -> None:
        """Initialize ConditionalEmbs. Block for 
        conditional embedding.

        Args:
            sample_length (int): length L of the samples of shape (B C L).
            input_length (int): length L' of the input.
            in_c (int): number of channels in the input tensor.
        """
        super(ConditionEmbedder, self).__init__()
        self.gamma_beta = GammaBeta(512, in_c)

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

        gamma, beta = self.gamma_beta(noise_scale)
        return gamma * x + beta
