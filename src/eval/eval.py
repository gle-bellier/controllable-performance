import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.cli import LightningCLI
import warnings


def main():
    cli = LightningCLI(pl.LightningModule,
                       pl.LightningDataModule,
                       parser_kwargs={
                           "parser_mode":
                           "omegaconf",
                           "default_config_files":
                           ["configs/urmp_conditional.yaml"]
                       },
                       run=False,
                       subclass_mode_model=True,
                       subclass_mode_data=True)

    state_dict = torch.load(
        "logs/exd_conditional/version_0/checkpoints/epoch=21299-step=468600.ckpt"
    )['state_dict']
    model = cli.model.load_state_dict(state_dict)


if __name__ == "__main__":
    main()