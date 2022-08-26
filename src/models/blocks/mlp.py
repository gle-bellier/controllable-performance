import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):

    def __init__(self,
                 mlp_dimensions: List[int],
                 non_linearity=nn.LeakyReLU(),
                 last_non_linearity=None) -> None:
        """Initialize Multi Layer Perceptron. 

        Args:
            mlp_dimensions (List[int]): list of hidden dimensions.
            non_linearity (callable), optional): activation function. Defaults to nn.LeakyReLU().
            last_non_linearity (callable, optional): activation function of the last layer. Defaults to None.
        """

        super().__init__()

        assert len(mlp_dimensions) > 0
        self.layers = nn.ModuleList([
            nn.Linear(mlp_dimensions[i - 1], mlp_dimensions[i])
            for i in range(1, len(mlp_dimensions))
        ])
        self.non_linearity = non_linearity
        self.last_non_linearity = last_non_linearity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.

        Args:
            x (torch.Tensor): input tensor of shape (..., H_in)
            where H_in is the first element of the mlp_dimensions list.

        Returns:
            torch.Tensor: output tensor of shape (..., H_out) where 
            H_out is the last element of the mlp_dimensions list.
        """
        for l in self.layers[:-1]:
            x = self.non_linearity(l(x))
        y = self.layers[-1](x)
        if self.last_non_linearity:
            y = self.non_linearity(y)
        return y
