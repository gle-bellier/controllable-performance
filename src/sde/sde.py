import torch
import numpy as np


class SDE():

    def __init__(self):
        self.t_max = 1
        self.t_min = 0

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def mean(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def perturb(self,
                x: torch.Tensor,
                t: torch.Tensor,
                noise=None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x)
        mean = self.mean(t)
        sigma = self.sigma(t)
        return mean * x + sigma * noise


class VpSdeSigmoid(SDE):

    def __init__(self) -> None:
        super().__init__()
        self.t_min = 0.0
        self.t_max = 1.0

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(22.31 * t - 18.42)**0.5

    def mean(self, t: torch.Tensor) -> torch.Tensor:
        return (1 - self.sigma(t)**2)**0.5

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return 22.31 * torch.sigmoid(22.31 * t - 18.42)

    def g(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta(t)**0.5


class VpSdeCos(SDE):

    def __init__(self) -> None:
        super().__init__()
        self.t_min = 0.007
        self.t_max = 1 - 0.007

    def sigma_inverse(self, sigma: torch.tensor) -> torch.Tensor:
        return 1 / np.pi * torch.acos(1 - 2 * sigma)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1 - torch.cos(np.pi * t))

    def mean(self, t: torch.Tensor) -> torch.Tensor:
        return (1 - self.sigma(t)**2)**0.5

    def sigma_derivative(self, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * np.pi * torch.sin(np.pi * t)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return 2 * self.sigma(t) * self.sigma_derivative(t) / (
            1 - self.sigma(t)**2)

    def g(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta(t)**0.5


class SubVpSdeCos(SDE):

    def __init__(self) -> None:
        super().__init__()
        self.t_min = 0.006
        self.t_max = 1 - 0.006

    def sigma_inverse(self, sigma: torch.tensor) -> torch.Tensor:
        return 1 / np.pi * torch.acos(1 - 2 * sigma)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1 - torch.cos(np.pi * t))

    def mean(self, t: torch.Tensor) -> torch.Tensor:
        return (1 - self.sigma(t))**0.5

    def sigma_from_mean_approx(self, mean_t, nu_t) -> torch.Tensor:
        return (1 - mean_t**2 - nu_t**2)

    def sigma_derivative(self, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * np.pi * torch.sin(np.pi * t)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_derivative(t) / (1 - self.sigma(t))

    def g(self, t: torch.Tensor) -> torch.Tensor:
        return (self.sigma_derivative(t) * self.sigma(t) *
                (2 - self.sigma(t)) / (1 - self.sigma(t)))**0.5