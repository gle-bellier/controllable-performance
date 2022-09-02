import torch
import torch.nn as nn
from samplers.sampler import Sampler
from sde.sde import Sde


class ODESampler(Sampler):

    def __init__(self,
                 sde: Sde,
                 model: nn.Module,
                 striding_type="linear",
                 thresholding=None,
                 quantile=None) -> None:
        super().__init__(sde, model, striding_type, thresholding, quantile)

        self.t_schedule = None
        self.sigma_schedule = None
        self.m_schedule = None
        self.beta_schedule = None
        self.g_schedule = None
        self.dt = None

    def striding(self, n_steps: int, t0=1.) -> None:
        """Compute the different schedules for sampling.
        Args:
            n_steps (int): number of steps to use during the sampling
            process.
            t0 (float): starting time of the diffusion process.
        """
        t_schedule = t0 * self.get_time_schedule(n_steps)
        self.t_schedule = (self.sde.t_max -
                           self.sde.t_min) * t_schedule + self.sde.t_min
        self.dt = self.t_schedule[1] - self.t_schedule[0]
        self.sigma_schedule = self.sde.sigma(self.t_schedule)
        self.m_schedule = self.sde.mean(self.t_schedule)
        self.beta_schedule = self.sde.beta(self.t_schedule)
        self.g_schedule = self.sde.g(self.t_schedule)

    def noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Noise input data x to the time step t.

        Args:
            x (torch.Tensor): data tensor of shape (B C L).
            t (torch.Tensor): time tensor of shape (B 1).

        Returns:
            torch.Tensor: noised tensor of shape (B C L).
        """
        return self.sde.mean(t) * x + self.sde.sigma(t) * torch.randn_like(x)

    def noise_step(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Noise input data x to the denoising step n.

        Args:
            x (torch.Tensor): data tensor of shape (B C L).
            t (torch.Tensor): time tensor of shape (B 1).

        Returns:
            torch.Tensor: noised tensor of shape (B C L).
        """
        return self.noise(x, self.t_schedule[n])

    def denoising(self, x: torch.Tensor, condition: torch.Tensor,
                  n: int) -> torch.Tensor:
        """Achieve denoising.

        Args:
            x (torch.Tensor): data tensor of shape (B C L).
            condition (torch.Tensor): condition tensor of shape (B C L).
            n (int): number of the step.

        Returns:
            torch.Tensor: denoised tensor at step n-1 of shape
            (B C L).
        """

        noise_scale = self.sigma_schedule[n] * torch.ones(
            condition.shape[0], 1, device=condition.device)

        return (1 + self.dt * self.beta_schedule[n] /
                2) * x - self.dt * self.g_schedule[n]**2 / (
                    2 * self.sigma_schedule[n]) * self.model(
                        x, condition, noise_scale)

    def jump_step(self, x: torch.Tensor,
                  condition: torch.Tensor) -> torch.Tensor:
        """Achieve last step of denoising (aka jump step).

        Args:
            x (torch.Tensor): data tensor of shape (B C L).
            condition (torch.Tensor): condition tensor of shape (B C L).


        Returns:
            torch.Tensor: denoised tensor of shape (B C L).
        """
        noise_scale = self.sigma_schedule[0] * torch.ones(
            condition.shape[0], 1, device=condition.device)
        return (x - self.sigma_schedule[0] * self.model(
            x, condition, noise_scale)) / self.sde.mean(self.t_schedule[0])
