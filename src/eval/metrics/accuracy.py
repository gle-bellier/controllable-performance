from typing import List
import torch
from numba import jit


class Midi_accuracy:

    def __init__(self, f0_threshold=.5, lo_threshold=.5):
        """Pitch accuracy estimating how accurate are expressive pitch contours according to 
        the unexpressive contours (the reference). Note to note mean frequency comparison.

        Args:
            f0_threshold (float, optional): fundamental frequency threshold (in semi tone). 
            Defaults to .5.
            lo_threshold (float, optional): loudness threshold (in dB). Defaults to .5.
        """
        self.f0_threshold = f0_threshold
        self.lo_threshold = lo_threshold

    def forward(self, cond_f0: torch.Tensor, gen_f0: torch.Tensor,
                cond_lo: torch.Tensor, gen_lo: torch.Tensor,
                mask: torch.Tensor) -> List[torch.Tensor]:
        """Compute accuracy for pitch (in semitone) and loudness (in dB). 

        Args:
            cond_f0 (torch.Tensor): fundamental frequency contour of shape
            (B L) used as condition to the generation.
            gen_f0 (torch.Tensor): generated fundamental frequency contours 
            of shape (B L).
            cond_lo (torch.Tensor): loudness contours of shape (B L) used
            as condition for the generation. 
            gen_lo (torch.Tensor): generated loudness contours of shape (B L).
            mask (torch.Tensor): mask of shape (B L N) where N is the number of 
            notes in the sample of length L.

        Returns:
            List[torch.Tensor]: list of the two accuracies (fundamental frequency
            and loudness).
        """

        # apply mask to the pitch contours

        mk_gen_f0 = mask * gen_f0
        mk_cond_f0 = mask * cond_f0

        # apply mask to the loudness contours
        mk_gen_lo = mask * gen_lo
        mk_cond_lo = mask * cond_lo

        accuracy_pitch = self.__contour_accuracy(mk_gen_f0, mk_cond_f0, mask,
                                                 self.f0_threshold)

        accuracy_lo = self.__contour_accuracy(mk_gen_lo, mk_cond_lo, mask,
                                              self.lo_threshold)

        return accuracy_pitch, accuracy_lo

    def __contour_accuracy(self, mk_gen: torch.Tensor, mk_target: torch.Tensor,
                           mask: torch.Tensor,
                           threshold: float) -> torch.Tensor:
        """Compute contour accuracy.

        Args:
            mk_gen (torch.Tensor): masked generated contours of shape (B L N). 
            mk_target (torch.Tensor): masked target contours of shape (B L N). 
            mask (torch.Tensor): mask of shape (B L N).
            threshold (float): threshold for the accuracy. 

        Returns:
            torch.Tensor: accuracy of shape (1,)
        """
        # compute the means for each notes for both contours

        mean_gen = torch.mean(mk_gen,
                              dim=-2) / (torch.mean(mask, dim=-2) + 1e-6)
        mean_target = torch.mean(mk_target,
                                 dim=-2) / (torch.mean(mask, dim=-2) + 1e-6)

        # compute the difference between means

        diff = torch.abs(mean_gen - mean_target) > threshold
        accuracy = 1 - torch.mean(diff.float())

        return accuracy
