import pytorch_lightning as pl
import torch
import torchaudio
from pytorch_lightning.utilities.cli import LightningCLI
from einops import rearrange

import matplotlib.pyplot as plt
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

        # prepare condition
        f0 = torch.Tensor(f0).reshape(1, 1, -1)
        lo = torch.Tensor(lo).reshape(1, 1, -1)
        mask = torch.from_numpy(mask)
        condition = torch.cat([f0, lo], -2).cuda()

        # conditional sampling
        contours, audio = cli.model.sample(condition)

        # rearrange and put on CPU
        contours = contours.cpu().detach().numpy()
        condition = condition.cpu().detach().numpy()
        # split generated contours
        g_f0, g_lo = rearrange(contours, "b c l -> c b l", c=2)
        # split unexpressive contours (from MIDI)
        u_f0, u_lo = rearrange(condition, "b c l -> c b l", c=2)

        # compute accuracy

        accuracy = metric.forward(u_f0, g_f0, u_lo, g_lo, mask)
        print(f"Accuracy: {accuracy}")

        # plot contours
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
        ax1.plot(u_f0.squeeze(), label="midi")
        ax1.plot(g_f0.squeeze(), label="gen")
        ax2.plot(u_lo.squeeze(), label="midi")
        ax2.plot(g_lo.squeeze(), label="gen")
        plt.legend()
        plt.savefig(f"generated/figures/exd_sample-{i_sample}.png")

        # save audio
        torchaudio.save(f"generated/audio/exd_sample-{i_sample}.wav",
                        audio.cpu(), 16000)
        plt.clf()

        i_sample += 1


if __name__ == "__main__":
    MIDI_FILE = "data/midi/midi_files/jupiter.mid"
    main(midi_path=MIDI_FILE)