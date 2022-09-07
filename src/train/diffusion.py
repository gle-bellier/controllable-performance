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
                 conditional_rate=0.3,
                 sample_length=1024) -> None:
        """Initialize denoising diffusion model.

        Args:
            data_processor (ContoursProcessor): dataprocessor used to process 
            the contours.
            transform (ConditionTransform): transformation applied to the condition
            signal.
            sampler (Sampler): denoising diffusion sampler.
            learning_rate (float): training learning rate. 
            conditional_rate (float): probability p of training the model without the condition 
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

    def sample(self, condition: torch.Tensor) -> Tuple[torch.Tensor]:
        """Sample new contours.

        Args:
            condition (torch.Tensor): conditioning contours
            of shape (B C L).

        Returns:
            Tuple[torch.Tensor]: sampled contours of shape (B C L)
            and audio generated of shape (B L*sampling_rate).
        """

        # apply preprocessing
        condition = self.P(condition)

        sample = self.sampler.sample(condition, n_steps=100)

        # apply inverse processing
        sample = (self.P - 1)(sample)

        # get audio
        audio = self.P.contours_to_wav(sample)

        return sample, audio

    def generate(self, n_samples: int) -> dict:
        """Generate contours from MIDI and expressive contours.

        Args:
            n_samples (int): number of contours to generate in 
            the two categories (from MIDI and from expressive 
            contours).

        Returns:
            dict: {
            "contours": (midi_contours, expressive_contours),
            "audio": (midi_audio, expressive_audio)
        } 
        """

        # get midi contours as condition
        midi = self.get_midi(n_samples, self.sample_length)
        midi = self.T(midi)
        midi = self.P(midi)

        # get expressive contours as condition
        expressive = self.get_expressive(n_samples.self.sample_length)
        expressive = self.P(expressive)

        # get samples
        midi_contours, midi_audio = self.sample(midi)
        expressive_contours, expressive_audio = self.sample(expressive)

        return {
            "contours": (midi_contours, expressive_contours),
            "audio": (midi_audio, expressive_audio)
        }

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

        loss = self(contours)
        # log the training loss
        self.log("train_loss", loss)

        self.train_step_idx += 1

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

        loss = self(contours)
        # log the training loss
        self.log("val_loss", loss)

        self.val_step_idx += 1

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
                            contours, self.val_step_idx)
        listen_batch_contours(self.logger.experiment, expressive_audio,
                              self.val_step_idx)
