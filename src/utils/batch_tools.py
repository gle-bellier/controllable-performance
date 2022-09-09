import torch

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt


def plot_batch_contours(writer: SummaryWriter, sample: torch.Tensor,
                        condition: torch.Tensor, epoch_id: int,
                        mode: str) -> None:
    """Plot random contours of the batch of contours to the writer

    Args:
        writer (SummaryWriter): experiment writer.
        sample (torch.Tensor): sample of shape (B C L).
        condition (torch.Tensor): condition of shape (B C L).
        epoch_id (int): id of the epoch.
        mode (str): train or val or sample (during validation).
    """

    # select random element of the batch
    s = torch.clone(sample).detach()
    c = torch.clone(condition).detach()
    batch_size = sample.shape[0]
    index = np.random.randint(0, batch_size)

    sample_f0, sample_lo = s[index].cpu()
    condition_f0, condition_lo = c[index].cpu()

    plt.plot(sample_f0, label="sample")
    plt.plot(condition_f0, label="condition")
    plt.legend()
    writer.add_figure(f"{mode}/f0", plt.gcf(), epoch_id)
    plt.plot(sample_lo, label="sample")
    plt.plot(condition_lo, label="condition")
    plt.legend()
    writer.add_figure(f"{mode}/lo", plt.gcf(), epoch_id)


def listen_batch_contours(writer: SummaryWriter, audio: torch.Tensor,
                          epoch_id: int) -> None:
    """Listen to random contours of the batch of contours to the writer

    Args:
        writer (SummaryWriter): experiment writer.
        audio (torch.Tensor): audio batch of shape (B C L).
        epoch_id (int): id of the epoch.
    """

    # select random element of the batch
    batch_size = audio.shape[0]
    index = np.random.randint(0, batch_size)

    audio_sample = audio[index].reshape(1, -1, 1)
    audio_sample = audio_sample.detach().cpu().numpy()

    writer.add_audio("generated", audio_sample, epoch_id, sample_rate=16000)