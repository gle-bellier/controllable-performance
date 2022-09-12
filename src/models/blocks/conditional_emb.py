import torch
import torch.nn as nn
from models.blocks.gamma_beta import GammaBeta


class ConditionEmbedder(nn.Module):

    def __init__(self, sample_length: int, input_length: int, in_c: int,
                 conditional: bool, activation: callable) -> None:
        """Initialize ConditionalEmbs. Block for 
        conditional embedding.

        Args:
            sample_length (int): length L of the samples of shape (B C L).
            in_c (int): number of channels in the input tensor.
            conditional (bool): if set to True then conditional contours are taken 
            into account.
            activation (callable): activation function.
        """
        super(ConditionEmbedder, self).__init__()
        self.gamma_beta = GammaBeta(512, in_c)

        self.conditional = conditional
        if self.conditional:
            self.lin_contours = nn.Sequential(
                nn.Linear(sample_length, 512),
                activation(),
            )
            self.conv = nn.Sequential(
                nn.Conv1d(3, 1, kernel_size=3, padding=1), activation())

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
            # (B N) -> (B 1 N)
            noise_scale = noise_scale.unsqueeze(-2)
            # (B 2 sl) -> (B 2 N)
            contours = self.lin_contours(contours)
            # condition of shape (B 3 N)
            condition = torch.cat([contours, noise_scale], -2)
            # (B 3 N) -> (B N)
            condition = self.conv(condition).squeeze(-2)
        else:
            condition = noise_scale

        gamma, beta = self.gamma_beta(condition)
        res = gamma * x + beta

        return res + x
