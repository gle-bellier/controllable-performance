import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI

from train.diffusion import Diffusion

cli = LightningCLI(Diffusion,
                   pl.LightningDataModule,
                   run=False,
                   subclass_mode_model=True,
                   subclass_mode_data=True)

cli.trainer.fit(cli.model, cli.datamodule)