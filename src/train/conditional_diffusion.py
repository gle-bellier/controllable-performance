import torch
import torch.nn.functional as F

from data.dataprocessor.dataprocessor import ContoursProcessor
from data.transforms.transform import ConditionTransform
from samplers.sampler import Sampler

from train.diffusion import Diffusion


class ConditionalDiffusion(Diffusion):

    def __init__(self,
                 data_processor: ContoursProcessor,
                 transform: ConditionTransform,
                 sampler: Sampler,
                 conditional_rate: float,
                 learning_rate: float,
                 sample_length=1024) -> None:
        """Initialize denoising diffusion model.

        Args:
            data_processor (ContoursProcessor): dataprocessor used to process 
            the contours.
            transform (ConditionTransform): transformation applied to the condition
            signal.
            sampler (Sampler): denoising diffusion sampler.
            conditional_rate (float): rate of conditional training, e.g. 0.7 means the model
            is trained with conditioning contours with probability 0.7 and trained without 
            condition with probability 0.3.
            learning_rate (float): training learning rate. 
            sample_length (int, optional): length L of contours of shape (B C L).
            Defaults to 1024.
        """
        super().__init__(data_processor, transform, sampler, learning_rate,
                         sample_length)
        self.conditional_rate = conditional_rate

    def forward(self, contours: torch.Tensor) -> torch.Tensor:
        """Compute denoising diffusion model forward pass.

        Args:
            contours (torch.Tensor): input conditioning contours
            of shape (B C L).

        Returns:
            torch.Tensor: loss.
        """

        if torch.rand(1) > self.conditional_rate:
            # unconditional generation
            # with "mean contours" ie approx nul contours
            condition = torch.zeros_like(contours)
        else:
            # condition generation
            # condition = transformed (blurred/filtered) contours
            condition = self.T(contours)

        # preprocess the data
        contours = self.P(contours)
        condition = self.P(condition)

        # create time batch
        batch_size = contours.shape[0]
        t = torch.rand(batch_size, 1, 1, device=contours.device)

        # ensure t in [t_min, t_max)
        t = (self.sde.t_max - self.sde.t_min) * t + self.sde.t_min

        # sample noise z
        z = torch.randn_like(contours, device=contours.device)

        # noise contours and predict injected noise
        contours_t = self.sde.perturb(contours, t, z)
        noise_scale = self.sde.sigma(t).squeeze(-1)
        z_hat = self.model(contours_t, condition, noise_scale)

        loss = F.mse_loss(z, z_hat)

        return loss
