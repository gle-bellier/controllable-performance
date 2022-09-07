from typing import Union, Tuple
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import librosa as li
import numpy as np
import torch
import crepe
import pickle
from os import walk


class Extractor:
    """Contours extraction tools 
    """

    def __init__(self, sr=16000, block_size=160, ddsp=None) -> None:
        """Initialize extractor tool.
        Args:
            sr (int, optional): sampling rate. Defaults to 16000.
            block_size (int, optional): window size for f0 and loudness computing. Defaults to 160.
            ddsp ([type], optional): ddsp module used for reconstruction. Defaults to None.
        """

        self.sr = sr
        self.block_size = block_size
        self.ddsp = ddsp

    def extract_f0_lo(self, path: str) -> Union[torch.Tensor, torch.Tensor]:

        audio, fs = li.load(path, sr=self.sr)

        print(audio.shape)
        lo = self.extract_lo(audio, self.sr, self.block_size)
        f0 = self.extract_f0(audio, self.sr, self.block_size)

        return f0, lo

    def extract_lo(self, signal, sampling_rate, block_size, n_fft=2048):
        S = li.stft(
            signal,
            n_fft=n_fft,
            hop_length=block_size,
            win_length=n_fft,
            center=True,
        )
        S = np.log(abs(S) + 1e-7)
        f = li.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
        a_weight = li.A_weighting(f)

        S = S + a_weight.reshape(-1, 1)

        S = np.mean(S, 0)[..., :-1]

        return S

    def extract_f0(self, signal, sampling_rate, block_size):
        f0 = crepe.predict(
            signal,
            sampling_rate,
            step_size=int(1000 * block_size / sampling_rate),
            verbose=0,
            center=True,
            viterbi=True,
        )
        return f0[1].reshape(-1)[:-1]

    def export(self, data: dict, path: str) -> None:
        """Export data into pickle file
        Args:
            data (dict): data dictionary
            path (str): path to the file
        """
        with open(path, "ab+") as file_out:
            pickle.dump(data, file_out)


class Builder:

    def __init__(self, audio_path, contours_path):
        self.audio_path = audio_path
        self.contours_path = contours_path

        self.extractor = Extractor()

    def get_samples_names(self):
        return next(walk(self.audio_path), (None, None, []))[2]

    def run(self):
        print("Processing files in ", self.audio_path)
        samples = self.get_samples_names()
        for sample in samples:
            print("<- Extracting contours from  ", sample)
            f0, lo = self.extractor.extract_f0_lo(self.audio_path + sample)
            data = {"f0": f0, "lo": lo}
            filename = sample[:-4] + ".pickle"
            print("-> Exporting ", filename)
            self.extractor.export(data, self.contours_path + filename)


if __name__ == "__main__":

    b = Builder("data/audio/processed/", "data/contours/expressive/extended/")
    b.run()