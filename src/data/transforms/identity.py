import torch
from data.transforms.transform import ConditionTransform


class IdentityTransform(ConditionTransform):
    """Conditioning contours transform method.
    """

    def __init__(self) -> None:
        """Initialize ConditionTransform.
        """
        super().__init__()

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transform to a input tensor.

        Args:
            x (torch.Tensor): conditioning tensor of shape
            (B C L).
            
        Returns:
            torch.Tensor: modified condition.
        """
        return x