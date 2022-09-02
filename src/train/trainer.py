import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI

from train.diffusion import Diffusion
import warnings

warnings.filterwarnings("ignore")

cli = LightningCLI(Diffusion,
                   pl.LightningDataModule,
                   parser_kwargs={
                       "parser_mode": "omegaconf",
                       "default_config_files": ["configs/efficient_unet.yaml"]
                   },
                   run=False,
                   subclass_mode_model=True,
                   subclass_mode_data=True)

cli.trainer.fit(cli.model, cli.datamodule)