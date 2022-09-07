import torch
import torch.nn as nn

from models.blocks.conditional_emb import ConditionEmbedder
from models.blocks.resnet_block import ResNetBlock


class DownBlock(nn.Module):

    def __init__(self, sample_length: int, input_length: int, in_c: int,
                 out_c: int, stride: int, num_resnets: int,
                 conditional: bool) -> None:
        """Initialize DownBlock.

        Args:
            sample_length (int): length L of the contours of shape (B 2 L'). 
            input_length (int): length L of the input of shape (B C L). 
            in_c (int): number of input channels in convolution.
            out_c (int): number of ouput channels in convolution.
            stride (int): stride of the convolution.
            num_resnets (int): number of resnets in the downblock.
            conditional (bool): if set to True then conditional downsampling is 
            computed else unconditional downsampling.
        """
        super(DownBlock, self).__init__()
        self.conditional = conditional

        self.dw = nn.Sequential(
            nn.Conv1d(in_channels=in_c,
                      out_channels=out_c,
                      kernel_size=3,
                      stride=stride,
                      padding=self.get_padding(3, stride, 1)), nn.LeakyReLU())

        self.embedder = ConditionEmbedder(sample_length=sample_length,
                                          input_length=input_length // stride,
                                          in_c=out_c,
                                          conditional=conditional)
        self.residual = nn.Sequential(*[
            ResNetBlock(input_length=input_length // 2, in_c=out_c)
            for _ in range(num_resnets)
        ])

    def get_padding(self, kernel_size: int, stride: int, dilation: int) -> int:
        """Return size of the padding needed.
        Args:
            kernel_size (int): kernel size of the convolutional layer
            stride (int): stride of the convolutional layer
            dilation (int): dilation of the convolutional layer
        Returns:
            int: padding
        """
        full_kernel = (kernel_size - 1) * dilation + 1
        return full_kernel // 2

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
