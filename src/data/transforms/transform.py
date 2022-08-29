import torch


class ConditionTransform:
    """Conditioning contours transform method.
    """

    def __init__(self) -> None:
        """Initialize ConditionTransform.
        """
        pass

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transform to a input tensor.

        Args:
            x (torch.Tensor): conditioning tensor of shape
            (B C L).
            
            NotImplementedError: error if not implemented.

        Returns:
            torch.Tensor: modified condition.
        """
        raise NotImplementedError