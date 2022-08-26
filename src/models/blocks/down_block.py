import torch
import torch.nn as nn

from models.blocks.conditional_emb import ConditionalEmbs
from models.blocks.resnet_block import ResNetBlock


class DownBlock(nn.Module):

    def __init__(self, n_sample: int, in_c: int, out_c: int, stride: int,
                 num_resnets: int) -> None:
        """Initialize DownBlock.

        Args:
            in_c (int): number of input channels in convolution.
            out_c (int): number of ouput channels in convolution.
            stride (int): stride of the convolution.
            num_resnets (int): number of resnets in the downblock.
        """
        super(DownBlock, self).__init__()

        self.dw = nn.Conv1d(in_channels=in_c,
                            out_channels=out_c,
                            kernel_size=3,
                            stride=stride,
                            padding=self.get_padding(3, stride, 1))

        self.noise_emb = ConditionalEmbs(n_sample=n_sample, in_c=out_c)
        self.residual = nn.Sequential(
            *[ResNetBlock(out_c, n_sample) for _ in range(num_resnets)])

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

    def forward(self, x: torch.Tensor,
                noise_scale: torch.Tensor) -> torch.Tensor:
        """Compute pass forward.

        Args:
            x (torch.Tensor): input tensor of shape (B C_in L_in).
            noise_scale (torch.Tensor): noise scale of shape (B, 1).

        Returns:
            torch.Tensor: output tensor of shape (B C_out L_out).
        """

        x = self.dw(x)
        x = self.noise_emb(x, noise_scale)
        return self.residual(x)


d = DownBlock(1024, 2, 4, 4, 2)

print(d(torch.randn(13, 2, 1024)))