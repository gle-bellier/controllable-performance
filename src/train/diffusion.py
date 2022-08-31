import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, OrderedDict, Tuple, List
import pytorch_lightning as pl

from data.dataprocessor.dataprocessor import ContoursProcessor
from data.transforms.transform import ConditionTransform
from sde.sde import Sde


class Diffusion(pl.LightningModule):

    def __init__(self,
                 model: nn.Module,
                 data_processor: ContoursProcessor,
                 transform: ConditionTransform,
                 sde: Sde,
                 sample_length=1024) -> None:
        super().__init__()

        self.model = model
        self.P = data_processor
        self.T = transform
        self.sde = sde
        self.sample_length = sample_length

        self.train_step_idx = 0
        self.val_step_idx = 0

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers.
        Returns:
            torch.optim.Optimizer: optimizer.
        """
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=1e-3,
                                      weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=150000,
                                                    gamma=0.5)
        return [optimizer], [scheduler]

    def forward(self, contours: torch.Tensor) -> torch.Tensor:

        # condition = transformed (blurred/filtered) contours
        condition = self.T(contours)

        # preprocess the data
        contours = self.P(contours)
        condition = self.P(condition)

        # create time batch
        batch_size = contours.shape[0]
        t = torch.rand(batch_size, 1, device=contours.device)

        # ensure t in [t_min, t_max)
        t = (self.sde.t_max - self.sde.t_min) * t + self.sde.t_min

        # sample noise z
        z = torch.randn_like(contours, device=contours.device)

        # noise contours and predict injected noise
        contours_t = self.sde.perturb(contours, t, z)
        z_hat = self.model(contours_t, condition, self.sde.sigma(t))

        loss = F.mse_loss(z, z_hat)

        return loss

    def sample(self, condition: torch.Tensor) -> Tuple[torch.Tensor]:
        sample = self.sampler.sample(condition)

        # apply inverse processing
        sample = (self.P - 1)(sample)

        # get audio
        audio = self.P.contours_to_wav(sample)

        return sample, audio

    def generate(self, n_samples: int) -> torch.Tensor:

        self.model.eval()

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

        loss = self(contours)
        # log the training loss
        self.log("train_loss", loss)

        self.train_step_idx += 1

        return {"loss": loss}

    def validation_step(self, contours: torch.Tensor,
                        batch_idx: int) -> OrderedDict:

        loss = self(contours)
        # log the training loss
        self.log("val_loss", loss)

        self.val_step_idx += 1

        return {"loss": loss}
