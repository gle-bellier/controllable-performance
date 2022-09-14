import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
import warnings

warnings.filterwarnings("ignore")


def main():
    cli = LightningCLI(pl.LightningModule,
                       pl.LightningDataModule,
                       parser_kwargs={
                           "parser_mode": "omegaconf",
                           "default_config_files":
                           ["configs/efficient_unet.yaml"]
                       },
                       run=False,
                       subclass_mode_model=True,
                       subclass_mode_data=True)

    cli.trainer.fit(cli.model, cli.datamodule)


if __name__ == "__main__":
    main()