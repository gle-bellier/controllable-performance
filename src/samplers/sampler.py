import torch
import torch.nn as nn
from sde.sde import Sde
from tqdm import tqdm
import numpy as np


class Sampler:

    def __init__(self,
                 sde: Sde,
                 model: nn.Module,
                 striding_type="linear",
                 thresholding=None,
                 quantile=None) -> None:
        """Initialize sampler.

        Args:
            sde (SDE): SDE used in the diffusion process.
            model (nn.Module): pretrained model.
            striding (str): either "linear" or "quadratic". Defaults to linear.
            thresholding (str, optional): type of thresholding applied. Available
            "static", "dynamic", None for no thresholding. Defaults to None.
            quantile (float, optional): quantile in case of dynamic thresholding.
            Defaults to None.
        """
        self.sde = sde
        self.model = model
        self.striding_type = striding_type
        self.thresholding = thresholding
        self.quantile = quantile

    def _to_numpy(self, tens: torch.Tensor) -> np.ndarray:
        """Convert tensor of GPU to numpy array on cpu.

        Args:
            tens (torch.Tensor): tensor of interest on GPU
            of CPU.

        Returns:
            np.ndarray: numpy array on CPU.
        """
        a = np.array(tens.cpu().numpy(), dtype=np.float64)
        return a

    def to_tensor(self, array: np.ndarray, device) -> torch.Tensor:
        """Convert numpy array on CPU to tensor of GPU.

        Args:
            array (np.ndarray): numpy array of interest on CPU.
            device: target device.

        Returns:
            torch.Tensor: torch tensor on GPU.
        """
        return torch.tensor(array, dtype=torch.float64).to(device=device)

    def striding(self, n_steps: int, t0=1.) -> None:
        """Compute the different schedules for sampling.
        Args:
            n_steps (int): number of steps to use during the sampling
            process.
            t0 (float): starting time of the diffusion process. Defaults to 1..

        Raises:
            NotImplementedError: raise error.
        """
        raise NotImplementedError

    def noise_step(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Noise input data x to the denoising step n.

        Args:
            x (torch.Tensor): data tensor of shape (B C L).
            t (torch.Tensor): time tensor of shape (B 1).

        Raises:
            NotImplementedError: raise error.

        Returns:
            torch.Tensor: noised tensor of shape (B C L).
        """

        raise NotImplementedError

    def denoising_step(self, x: torch.Tensor, condition: torch.Tensor,
                       n: int) -> torch.Tensor:
        """Achieve denoising step i.e. denoising of the input
        and thresholding if necessary.

        Args:
            x (torch.Tensor): data tensor of shape (B C L).
            condition (torch.Tensor): condition tensor of shape (B C L).
            n (int): number of the step.

        Raises:
            NotImplementedError: raise error.

        Returns:
            torch.Tensor: denoised tensor at step n-1 of shape
            (B C L).
        """
        x = self.denoising(x, condition, n)

        if self.thresholding == None:
            return x
        elif self.thresholding == "static":
            return self.static_thresholding(x)
        elif self.thresholding == "dynamic":
            return self.dynamic_thresholding(x, self.quantile)
        else:
            return x

    def denoising(self, x: torch.Tensor, condition: torch.Tensor,
                  n: int) -> torch.Tensor:
        """Achieve denoising.

        Args:
            x (torch.Tensor): data tensor of shape (B C L).
            condition (torch.Tensor): condition tensor of shape (B C L).
            n (int): number of the step.

        Raises:
            NotImplementedError: raise error.

        Returns:
            torch.Tensor: denoised tensor at step n-1 of shape
            (B C L).
        """
        raise NotImplementedError

    def jump_step(self, x: torch.Tensor,
                  condition: torch.Tensor) -> torch.Tensor:
        """Achieve last step of denoising (aka jump step).

        Args:
            x (torch.Tensor): data tensor of shape (B C L).
            condition (torch.Tensor): condition tensor of shape (B C L).

        Raises:
            NotImplementedError: raise error.

        Returns:
            torch.Tensor: denoised tensor of shape (B C L).
        """
        raise NotImplementedError

    def sample(
        self,
        condition: torch.Tensor,
        n_steps: int,
    ) -> torch.Tensor:
        """Compute the sampling process.
        Args:
            condition (torch.Tensor): condition tensor of shape (B C L).
            n_steps (int): number of steps to use for sampling.

        Raises:
            NotImplementedError: raise error.

        Returns:
            torch.Tensor: generated tensor of shape (B C L).
        """
        self.striding(n_steps)
        x = torch.randn_like(condition, device=condition.device)

        with torch.no_grad():

            for n in tqdm(range(n_steps - 1, 0, -1)):

                x = self.denoising_step(x, condition, n)

            x = self.jump_step(x, condition)

        return x

    def __quantile_value(self, x: torch.Tensor,
                         quantile: float) -> torch.Tensor:
        """Compute the quantile value.

        Args:
            x (torch.Tensor): input tensor of shape (B C L).
            quantile (float): quantile.

        Returns:
            torch.Tensor: quantile value tensor of shape (B 1).
        """

        s = torch.sort(x.abs())[0]
        index = int(x.shape[-1] * quantile) - 1
        qv = s[:, :, index:index + 1]
        return qv

    def dynamic_thresholding(self, x: torch.Tensor,
                             quantile: float) -> torch.Tensor:
        """Compute dynamic thresholding as in the Imagen article
        (https://arxiv.org/abs/2205.11487).

        Args:
            x (torch.Tensor): input tensor of shape (B C*L).
            quantile (float): quantile (100*percentile).

        Returns:
            torch.Tensor: dynamically clipped tensor of shape (B C*L).
        """

        qv = self.__quantile_value(x, quantile)
        # if qv > 1 then clip else keep
        qv = torch.maximum(qv, torch.ones_like(qv))
        # hard clip for all points outside
        x = torch.clip(x, -qv, qv)
        # divide to rescale to [-1, 1]
        x /= torch.abs(qv)

        return x

    def get_time_schedule(self, n_steps: int) -> torch.Tensor:
        """Get time scheduling for the sampling process.

        Args:
            num_steps (int): number of step to use for sampling.

        Returns:
            torch.Tensor: time schedule of shape (n_steps,)
        """

        if self.striding_type == "quadratic":
            return torch.linspace(0, 1, n_steps)**2
        else:
            return torch.linspace(0, 1, n_steps)

    def static_thresholding(self, x: torch.Tensor) -> torch.Tensor:
        """Compute static thresholding.

        Args:
            x (torch.Tensor): input tensor of shape (B C*L).

        Returns:
            torch.Tensor: clipped tensor of shape (B C*L).
        """
        return torch.clip(x, -1, 1)
