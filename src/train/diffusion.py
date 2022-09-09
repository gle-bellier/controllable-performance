import torch
import torch.nn.functional as F

from typing import Any, OrderedDict, Tuple, List
import pytorch_lightning as pl

from data.dataprocessor.dataprocessor import ContoursProcessor
from data.transforms.transform import ConditionTransform
from samplers.sampler import Sampler

from utils.batch_tools import plot_batch_contours, listen_batch_contours


class Diffusion(pl.LightningModule):

    def __init__(self,
                 data_processor: ContoursProcessor,
                 transform: ConditionTransform,
                 sampler: Sampler,
                 learning_rate: float,
                 conditional_rate=0.5,
                 sample_length=1024) -> None:
        """Initialize denoising diffusion model.

        Args:
            data_processor (ContoursProcessor): dataprocessor used to process 
            the contours.
            transform (ConditionTransform): transformation applied to the condition
            signal.
            sampler (Sampler): denoising diffusion sampler.
            learning_rate (float): training learning rate. 
            conditional_rate (float): probability p of training the model with the condition 
            (in the conditional training scenario, for unconditional training it has no effet 
            on the training).
            sample_length (int, optional): length L of contours of shape (B C L).
            Defaults to 1024.
        """
        super().__init__()

        self.P = data_processor
        self.T = transform

        self.sampler = sampler
        # get model and sde

        self.model = self.sampler.model
        self.sde = self.sampler.sde

        self.learning_rate = learning_rate
        self.conditional_rate = conditional_rate
        self.sample_length = sample_length

        self.train_step_idx = 0
        self.val_step_idx = 0

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer]:
        """Configure optimizers.
        Returns:
            torch.optim.Optimizer: optimizer.
            torch.optim._LRScheduler: scheduler.
        """
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=150000,
                                                    gamma=0.5)
        return [optimizer], [scheduler]

    def forward(self, contours: torch.Tensor) -> Tuple[torch.Tensor]:
        """Compute denoising diffusion model forward pass.

        Args:
            contours (torch.Tensor): input conditioning contours
            of shape (B C L).

        Returns:
            Tuple[torch.Tensor]: loss tensor and denoised contours 
            of shape (B C L).
            
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
        scaled_contours = self.P(contours)
        scaled_condition = self.P(condition)

        # create time batch
        batch_size = contours.shape[0]

        # make sure the batch covers the range [0, 1] uniformly with a random offset

        t = (torch.rand(1, device=scaled_contours.device) + torch.arange(
            batch_size, device=scaled_contours.device) / batch_size) % 1

        # ensure t in [t_min, t_max)
        t = (self.sde.t_max - self.sde.t_min) * t.reshape(-1, 1,
                                                          1) + self.sde.t_min

        # sample noise z
        z = torch.randn_like(scaled_contours, device=scaled_contours.device)

        # noise contours and predict injected noise
        contours_t = self.sde.perturb(scaled_contours, t, z)
        noise_scale = self.sde.sigma(t).squeeze(-1)

        # noise prediction
        z_hat = self.model(contours_t, scaled_condition, noise_scale)
        # compute loss
        loss = F.mse_loss(z, z_hat)

        # compute denoised contours
        contours_hat = self.sde.perturb_inv(contours_t, z_hat, t)
        # apply post processing
        contours_hat = (self.P - 1)(contours_hat)

        return loss, contours_hat

    def sample(self, contours: torch.Tensor) -> Tuple[torch.Tensor]:
        """Sample new contours.

        Args:
            condition (torch.Tensor): conditioning contours
            of shape (B C L).

        Returns:
            Tuple[torch.Tensor]: sampled contours of shape (B C L)
            and audio generated of shape (B L*sampling_rate).
        """

        # apply transform and then preprocessing
        condition = self.T(contours)
        condition = self.P(condition)

        sample = self.sampler.sample(condition, n_steps=100)

        # apply inverse processing
        sample = (self.P - 1)(sample)

        # get audio
        audio = self.P.contours_to_wav(sample)

        return sample, audio

    def training_step(self, contours: torch.Tensor,
                      batch_idx: int) -> OrderedDict:
        """Compute training step.

        Args:
            contours (torch.Tensor): expressive contours
            of shape (B C L) from the training dataset.
            batch_idx (int): index of the batch.

        Returns:
            OrderedDict: {"loss"}
        """

        loss, contours_hat = self(contours)
        # log the training loss
        self.log("train_loss", loss)

        self.train_step_idx += 1
        if self.train_step_idx % 20 == 0:
            plot_batch_contours(self.logger.experiment, contours, contours_hat,
                                self.train_step_idx, "train")

        return {"loss": loss}

    def validation_step(self, contours: torch.Tensor,
                        batch_idx: int) -> OrderedDict:
        """Compute validation step.

        Args:
            contours (torch.Tensor): expressive contours
            of shape (B C L) from the validation dataset.
            batch_idx (int): index of the batch.

        Returns:
            OrderedDict: {"loss"}
        """

        loss, contours_hat = self(contours)
        # log the training loss
        self.log("val_loss", loss)

        self.val_step_idx += 1
        if self.train_step_idx % 20 == 0:
            plot_batch_contours(self.logger.experiment, contours, contours_hat,
                                self.train_step_idx, "val")

        return {"loss": loss, "contours": contours}

    def validation_epoch_end(self, batch_parts: List[dict]) -> None:
        """Plot and listen to generated contours at the end of the validation 
        step.

        Args:
            batch_parts (List[dict]): results of the validation step.
        """

        # select last item expressive contours
        contours = batch_parts[-1]["contours"]

        # get samples
        expressive_contours, expressive_audio = self.sample(contours)

        # plot and listen to the results
        plot_batch_contours(self.logger.experiment, expressive_contours,
                            contours, self.val_step_idx, "sample")
        listen_batch_contours(self.logger.experiment, expressive_audio,
                              self.val_step_idx)
