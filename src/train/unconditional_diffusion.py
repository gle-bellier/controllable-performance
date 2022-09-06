import torch
import torch.nn.functional as F

from train.diffusion import Diffusion


class UnconditionalDiffusion(Diffusion):

    def forward(self, contours: torch.Tensor) -> torch.Tensor:
        """Compute denoising diffusion model forward pass.

        Args:
            contours (torch.Tensor): input conditioning contours
            of shape (B C L).

        Returns:
            torch.Tensor: loss.
        """

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
