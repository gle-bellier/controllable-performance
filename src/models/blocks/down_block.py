import torch
import torch.nn as nn

from models.blocks.conditional_emb import ConditionEmbedder
from models.blocks.resnet_block import ResNetBlock


class Downsampling(nn.Module):

    def __init__(self, factor: int, in_c: int, out_c: int,
                 activation: callable) -> None:
        """Initialize downsampling module.

        Args:
            factor (int): downsampling factor.
            in_c (int): number of input channels in convolution.
            out_c (int): number of output channels in convolution.
            activation (callable): activation function.
        """
        super().__init__()

        self.down = nn.AvgPool1d(factor)
        self.conv = nn.Conv1d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=3,
                              padding=1)

        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute downsampling.

        Args:
            x (torch.Tensor): input tensor of shape (B C L).

        Returns:
            torch.Tensor: output tensor of shape (B C L/factor)
        """

        x = self.down(x)
        x = self.conv(x)
        return self.activation(x)


class DownBlock(nn.Module):

    def __init__(self, sample_length: int, input_length: int, in_c: int,
                 out_c: int, factor: int, num_resnets: int, conditional: bool,
                 activation: callable) -> None:
        """Initialize DownBlock.

        Args:
            sample_length (int): length L of the contours of shape (B 2 L'). 
            input_length (int): length L of the input of shape (B C L). 
            in_c (int): number of input channels in convolution.
            out_c (int): number of ouput channels in convolution.
            factor (int): downsampling factor.
            num_resnets (int): number of resnets in the downblock.
            conditional (bool): if set to True then conditional downsampling is 
            computed else unconditional downsampling.
            activation (callable): activation function.
        """
        super(DownBlock, self).__init__()
        self.conditional = conditional

        self.dw = Downsampling(factor=factor,
                               in_c=in_c,
                               out_c=out_c,
                               activation=activation)

        self.embedder = ConditionEmbedder(sample_length=sample_length,
                                          input_length=input_length // factor,
                                          in_c=out_c,
                                          conditional=conditional,
                                          activation=activation)
        self.residual = nn.Sequential(*[
            ResNetBlock(input_length=input_length // factor,
                        in_c=out_c,
                        activation=activation) for _ in range(num_resnets)
        ])

    def forward(self, x: torch.Tensor, condition: torch.Tensor,
                noise_scale: torch.Tensor) -> torch.Tensor:
        """Compute pass forward.

        Args:
            x (torch.Tensor): input tensor of shape (B C_in L_in).
            condition (torch.Tensor): condition tensor of shape (B C L)
            with C=2, L=sample_length.
            noise_scale (torch.Tensor): noise scale of shape (B, 512).

        Returns:
            torch.Tensor: output tensor of shape (B C_out L_out = L_in // stride).
        """

        x = self.dw(x)
        x = self.embedder(x, condition, noise_scale)

        return self.residual(x)
