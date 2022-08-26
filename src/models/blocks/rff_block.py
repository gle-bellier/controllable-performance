import torch
import torch.nn as nn
import numpy as np

from .mlp import MLP


class RFFBlock(nn.Module):

    def __init__(self,
                 input_dim=1,
                 rff_dim=32,
                 mlp_hidden_dims: list = None) -> None:
        """Initialize RFF block.
        Args:
            input_dim (int, optional): number of input features. Defaults to 1.
            rff_dim (int, optional): needs to be half of desired output shape. Defaults to 32.
            mlp_hidden_dims (list, optional): list of the hidden size of the MLP if need to use one after RFF. Defaults to None.
        """
        super().__init__()
        self.RFF_freq = nn.Parameter(16 * torch.randn([input_dim, rff_dim]),
                                     requires_grad=False)
        if mlp_hidden_dims:
            assert len(mlp_hidden_dims) > 0
            mlp_hidden_dims.insert(0, 2 * rff_dim)
            self.MLP = MLP(mlp_dimensions=mlp_hidden_dims,
                           non_linearity=nn.ReLU(),
                           last_non_linearity=nn.ReLU())
        else:
            self.MLP = None

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute pass forward.

        Args:
            std_step (torch.Tensor): tensor of shape (B, H_in)
            where H_in is equal to input_dim.

        Returns:
            torch.Tensor: embedding of sigma of shape (B, H_out)
            where H_out = 2*rrf_out.
        """
        x = self._build_RFF_embedding(sigma)
        if self.MLP:
            x = self.MLP(x)
        return x

    def _build_RFF_embedding(self, sigma: torch.Tensor) -> torch.Tensor:
        """Build the RFF embedding of sigma.

        Args:
            sigma (torch.Tensor): sigma of shape (..., 1)

        Returns:
            torch.Tensor: table of shape (..., H_out) where 
            H_out = 2*rff_out.
        """

        freqs = self.RFF_freq
        table = 2 * np.pi * torch.einsum('...i,ij->...j', sigma, freqs)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=-1)
        return table
