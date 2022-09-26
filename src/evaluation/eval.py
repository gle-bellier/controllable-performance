import pytorch_lightning as pl
import torch
import torchaudio
from pytorch_lightning.utilities.cli import LightningCLI
from einops import rearrange

from data.midi.midi_reader import MidiReader

from evaluation.metrics.accuracy import Midi_accuracy


def main(midi_path: str) -> None:
    cli = LightningCLI(pl.LightningModule,
                       pl.LightningDataModule,
                       parser_kwargs={
                           "parser_mode": "omegaconf",
                           "default_config_files":
                           ["configs/exd_conditional.yaml"]
                       },
                       run=False,
                       subclass_mode_model=True,
                       subclass_mode_data=True)

    state_dict = torch.load(
        "logs/exd_conditional/version_0/checkpoints/epoch=21399-step=470800.ckpt"
    )['state_dict']
    cli.model.load_state_dict(state_dict)
    cli.model.cuda()
    cli.model.eval()

    # get MIDI
    midi_reader = MidiReader()
    # get MIDI accuracy metric
    metric = Midi_accuracy()

    # loop on MIDI samples
    i_sample = 0
    for sample in midi_reader.get_contours(path=midi_path):

        f0, lo, _, _, mask = sample
        f0 = torch.Tensor(f0).reshape(1, 1, -1)
        lo = torch.Tensor(lo).reshape(1, 1, -1)
        mask = torch.from_numpy(mask)

        condition = torch.cat([f0, lo], -2).cuda()

        contours, audio = cli.model.sample(condition)

        e_f0, e_lo = rearrange(condition, "b c l -> c b l", c=2)
        e_f0 = e_f0.cpu().detach().numpy()
        f0 = f0.cpu().detach().numpy()

        import matplotlib.pyplot as plt

        plt.plot(f0.squeeze())
        plt.plot(e_f0.squeeze())
        plt.savefig(f"generated/figures/exd_sample-{i_sample}.png")

        # save audio
        torchaudio.save(f"generated/audio/exd_sample-{i_sample}.wav",
                        audio.cpu(), 16000)
        plt.clf()

        i_sample += 1


if __name__ == "__main__":
    MIDI_FILE = "data/midi/midi_files/slow.mid"
    main(midi_path=MIDI_FILE)