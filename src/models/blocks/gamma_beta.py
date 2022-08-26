import torch
import torch.nn as nn
from typing import Tuple


class GammaBeta(nn.Module):

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Initialize Gamma Beta block for split embeddings. (like
        in FiLM layers).

        Args:
            input_dim (int): number of input features.
            output_dim (int): number of output features.
        """
        super().__init__()
        self.output_layer = nn.Linear(input_dim, 2 * output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Compute forward pass.

        Args:
            x (torch.Tensor): tensor of shape (..., H_in)
            where H_in is the number of input features.

        Returns:
            Tuple[torch.Tensor]: (gamma, beta) tuple with gamma 
            and beta being of shape (..., H_out) where H_out is the
            number of output features.
        """
        x = self.output_layer(x)
        x = input.unsqueeze(-1)
        gamma, beta = torch.chunk(x, 2, dim=1)
        return gamma, beta
