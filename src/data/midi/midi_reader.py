import pretty_midi
import numpy as np
from typing import Tuple

from data.midi.velocity_to_loudness import LoudnessTransform


class MidiReader:

    def __init__(self, sample_len=1024, sustain_prop=0.2, frame_rate=100):
        """Useful tool to extract contours from a MIDI file. Be careful in the choice of sample_len: it must
        be greater than the length of the longest silence.
        Args:
            sample_len (int, optional): length of the MIDI samples. Defaults to 2048.
            sustain_prop (float, optional): proportion of the note considered as transitions, e.g, 
            if set to 0.2 it means attack length is 10% of the note length and same for the release.
            Defaults to 0.2.
            frame_rate (int, optional): frame_rate (number of frame by second). Defaults to 100.
        """

        self.frame_rate = frame_rate
        self.sample_len = sample_len
        # smoothinh the mask to take into account only sustain part
        self.sustain_prop = sustain_prop / 2

        self.data = None

        MIDI_PATH = "data/midi/midi_files/midi_ref/"
        CONTOURS_PATH = "data/contours/expressive/extended/dataset.pickle"
        self.loudness_transform = LoudnessTransform(MIDI_PATH, CONTOURS_PATH)

    def __len__(self) -> int:
        return int(self.midi.get_end_time() * self.frame_rate)

    def vel_to_lo(self, vel: np.ndarray) -> np.ndarray:
        """Mapping velocity contours from the MIDI file to 
        their loudness equivalent using a prefitted QuantileTransformer.

        Args:
            vel (np.ndarray): velocity contour of shape (L,).

        Returns:
            np.ndarray: loudness contour of shape of shape (L,).
        """
        vel = vel.reshape(-1, 1)
        self.loudness_transform = self.loudness_transform.fit(vel)
        lo = self.loudness_transform.transform(vel)
        return lo.squeeze()

    def get_contours(self, path: str, f0_range=[20, 90]) -> Tuple[np.ndarray]:
        """Compute te pitch and velocity contours from a MIDI file, 
        onsets, offsets and note activation mask.

        Args:
            path (str): path to the MIDI file.
            f0_range(optional, List[int]): range [min, Max] for the fundamental
            frequency contours. Contours are clipped to this range.
        Yields:
            Iterator[Tuple[np.ndarray]]: Tuple of pitch contour of shape (L,),
            loudness contour of shape (L,), onsets and offsets of shape (L, )
            and note activation mask of shape (N, L) with N the number of MIDI
            notes in the sample and L is the sample length. 
        """

        self.midi = pretty_midi.PrettyMIDI(path)
        self.data = self.midi.instruments[0]

        notes = self.data.get_piano_roll(self.frame_rate)
        f0, lo = self.__extract_f0_lo(notes)
        # clip f0 to the range
        f0 = np.clip(f0, f0_range[0], f0_range[1])
        onsets, offsets, mask = self.__get_onsets_mask()

        assert len(f0) == len(lo) == len(onsets) == len(offsets) == len(
            mask
        ), f"Invalid : numbers of samples are different : f0 : {len(f0)}, lo : {len(lo)}, onsets : {len(onsets)}, offsets: {len(offsets)} , mask: {len(mask)}"
        for i in range(len(f0)):
            yield (f0[i], lo[i], onsets[i], offsets[i], mask[i])

    def __attack_release(self, start_vel, peak, end_vel, length):

        attack = np.linspace(start_vel, peak - 5, length // 4)
        sustain = np.linspace(peak - 5, peak - 1, (length + 1) // 4)
        decay = np.linspace(peak - 1, peak - 10, (length + 2) // 4)
        release = np.linspace(peak - 10, end_vel, (length + 3) // 4)

        return np.concatenate((attack, sustain, decay, release))

    def __lissage_f0(self, x):
        for i in range(1, len(x)):
            if x[i] == 0:
                x[i] = x[i - 1]
        for i in range(len(x) - 1, -1, -1):
            if x[i] == 0:
                x[i] = x[i + 1]

        return x

    def __lissage_lo(self, x):

        for i in range(1, len(x) - 1):

            if np.sum(x[i - 2:i + 2]) != 0 and x[i] == 0:
                x[i] = np.max(x[i - 2:i + 2])
        return x

    def get_adsr(self, vel):

        adsr = np.zeros_like(vel)

        end_vel = 0
        for current, next_note in zip(self.data.notes[:-1],
                                      self.data.notes[0:]):

            start = int(current.start * self.frame_rate)
            end = int(current.end * self.frame_rate)
            next_start = int(next_note.start * self.frame_rate)
            next_end = int(next_note.end * self.frame_rate)

            peak = np.max(vel[start:end])
            next_peak = np.max(vel[next_start:next_end])

            start_vel = end_vel
            end_vel = (peak + next_peak) / 4

            adsr[start:end] = self.__attack_release(start_vel, peak, end_vel,
                                                    end - start)
        return adsr

    def __extract_f0_lo(self, notes) -> Tuple[np.ndarray]:
        """Extract pitch and loudness in the piano roll for each frame.
        Args:
            notes ([type]): track piano roll.
        Returns:
            Tuple[np.ndarray]: pitch and loudness arrays of shape (L,).
        """
        f0 = np.argmax(notes, axis=0)
        # we need to ensure continuity of fundamental frequency
        # we do not want it to drop during silence so we copy the
        # previous pitch value

        vel = np.transpose(np.max(notes, axis=0))
        # remove 0 pitch/velocity during successive notes gap

        adsr = self.get_adsr(vel)

        f0 = self.__lissage_f0(f0)
        adsr = self.__lissage_lo(adsr)

        # we need to convert velocity to loudness
        lo = self.vel_to_lo(adsr)

        # split contours into chunks for each sample
        f0 = np.split(f0, np.arange(self.sample_len, len(f0), self.sample_len))
        lo = np.split(lo, np.arange(self.sample_len, len(lo), self.sample_len))

        # we do not take into account the last chunk that has not a size equals to sample_len
        return f0[:-1], lo[:-1]

    def __get_onsets_mask(self) -> Tuple[np.ndarray]:
        """Create onsets and offsets contours and mask for the note in the track.
        We do not take into account the last chunk that has not a size equals to 
        sample_len.
        Returns:
            Tuple[np.ndarray]: onsets, offset and corresponding mask
        """

        l_onsets = []
        l_offsets = []
        l_masks = []

        l = self.sample_len
        onsets = np.zeros(l)
        offsets = np.zeros(l)
        mask = []

        i_sample = 0

        for note in self.data.notes:
            m = np.ones_like(onsets)

            start = int(
                max(0,
                    note.start * self.frame_rate - i_sample * self.sample_len))
            end = int(min(len(self) - 1, note.end *
                          self.frame_rate)) - i_sample * self.sample_len

            smooth = int((end - start) * self.sustain_prop)

            if start < l and end < l:
                onsets[start] = 1
                offsets[end] = 1

                # update mask
                m[:start + smooth] -= 1
                m[max(0, end - smooth):] -= 1
                mask += [m]

            elif start < l and end > l:
                onsets[start] = 1

                # update mask
                m[:start + smooth] -= 1
                mask += [m]

                # add the mask to the list of masks
                # add onsets, offsets to the lists
                l_masks += [np.array(mask)]
                l_onsets += [onsets]
                l_offsets += [offsets]

                # reset onsets, offsets and mask

                onsets = np.zeros(l)
                offsets = np.zeros(l)
                mask = []
                i_sample += 1
                # create new mask and add the end of the current note
                # if the note is longer than the sample length we crop it
                # to the end of the next sample
                offsets[min(end, len(offsets) - 1)] = 1

                m = np.ones_like(onsets)
                end -= self.sample_len
                m[max(0, end - smooth):] -= 1
                mask += [m]
                m = np.ones_like(onsets)

            else:
                #nothing to update go to next sample
                # add the mask to the list of masks
                # add onsets, offsets to the lists
                l_masks += [np.array(mask)]
                l_onsets += [onsets]
                l_offsets += [offsets]

                # reset onsets, offsets and mask

                onsets = np.zeros(l)
                offsets = np.zeros(l)
                mask = []
                i_sample += 1

        return l_onsets, l_offsets, l_masks


if __name__ == "__main__":

    mr = MidiReader()
    results = next(mr.get_contours("data/midi/midi_files/jupiter.mid"))

    print(results)