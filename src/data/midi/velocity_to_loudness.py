from os import walk
import pretty_midi
from typing import List, Tuple
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer
from pickle import dump, load


class LoudnessTransform(BaseEstimator, TransformerMixin):

    def __init__(self, midi_path: str, contours_path: str) -> None:
        """Initialize LoudnessTransform.

        Args:
            midi_path (str): path to the folder containing all the 
            MIDI files of reference.
            contours_path (str): path to the contours file.
        """

        self.midi_path = midi_path
        self.contours_path = contours_path
        self.lo_scaler = None
        self.vel_scaler = None

    def fit(self, X: np.ndarray) -> object:
        """Fit scaler to the dataset. It is independent to the 
        input array since fitted on reference MIDI and contours 
        dataset.

        Args:
            X (np.ndarray): input vector of shape (L, 1).

        Returns:
            object: fitted transformer.
        """
        # get vel from MIDI files of reference
        vel = []
        for p in self.get_midi_files():
            vel += [self.get_velocity(self.midi_path + p)]
        vel = np.concatenate(vel).reshape(-1, 1)

        # get loudness from contours files of reference
        lo = np.array([])
        for c in self.read_from_pickle():
            lo = np.concatenate((lo, c["lo"]))
        lo = lo.reshape(-1, 1)

        lo_scaler = QuantileTransformer(output_distribution="normal")
        self.lo_scaler = lo_scaler.fit(lo)
        vel_scaler = QuantileTransformer(output_distribution="normal")
        self.vel_scaler = vel_scaler.fit(vel)
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Transform velocity contours to loudness contours.

        Args:
            X (np.ndarray): velocity contour of shape (L, 1).
            y (_type_, optional): useless vector. Defaults to None.

        Returns:
            np.ndarray: loudness contour of shape (L, 1).
        """

        # transform: velocity -> normal
        normal = self.vel_scaler.transform(X)
        # inverse transform: normal -> loudness
        lo = self.lo_scaler.inverse_transform(normal)
        return lo

    def inverse_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Inverse transform loudness contours to velocity contours.

        Args:
            X (np.ndarray): loudness contour of shape (L, 1).
            y (_type_, optional): useless vector. Defaults to None.

        Returns:
            np.ndarray: velocity contour of shape (L, 1).
        """

        # transform: loudness -> normal
        normal = self.lo_scaler.transform(X)
        # inverse transform: normal -> velocity
        vel = self.vel_scaler.inverse_transform(normal)
        return vel

    def get_midi_files(self) -> List[str]:
        """Get MIDI files from PATH.

        Returns:
            List[str]: list of the filenames.
        """
        return next(walk(self.midi_path), (None, None, []))[2]

    def get_velocity(self, path: str, frame_rate=100) -> Tuple[np.ndarray]:
        """Get velocity from MIDI file.

        Args:
            path (str): path to the MIDI file.
            frame_rate (int, optional): frame rate for extraction. Defaults to 100.

        Returns:
            Tuple[np.ndarray]: velocity contour of shape (L,).
        """

        midi = pretty_midi.PrettyMIDI(path)
        data = midi.instruments[0]

        notes = data.get_piano_roll(frame_rate)
        return np.transpose(np.max(notes, axis=0))

    def read_from_pickle(self):
        """Read elements from pickle file.

        Yields:
            dict[np.ndarray]: contour of shape (L,).
        """
        with open(self.contours_path, 'rb') as file:
            try:
                while True:
                    yield load(file)

            except EOFError:
                pass
