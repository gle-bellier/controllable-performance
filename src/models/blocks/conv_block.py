import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, input_length: int, in_c: int, out_c: int) -> None:
        """Initialize ConvBlock.

        Args:
            input_length (int): length L of the input of shape (B C L).
            in_c (int): number of input channels in the convolution.
            out_c (int): number of output channels in the convolution.
        """
        super(ConvBlock, self).__init__()
        self.input_length = input_length
        #self.ln = nn.LayerNorm([channels, input_length])

        self.gn = nn.GroupNorm(num_groups=1, num_channels=in_c)
        self.conv = nn.Conv1d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=3,
                              stride=1,
                              padding=self.get_padding(3, 1, 1))
        self.swish = nn.SiLU()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pass forward.

        Args:
            x (torch.Tensor): input tensor of shape (B C L).

        Returns:
            torch.Tensor: output tensor of shape (B C L).
        """
        x = self.gn(x)
        x = self.swish(x)
        return self.conv(x)
