import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.cli import LightningCLI

from data.midi.midi_reader import MidiReader


def main(midi_path: str) -> None:
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

    # get MIDI
    midi_reader = MidiReader()
    f0, lo, _, _, _ = midi_reader.get_contours(path=midi_path)


if __name__ == "__main__":
    MIDI_FILE = "data/midi/midi_files/jupiter.mid"
    main(midi_path=MIDI_FILE)