import torch
from typing import List
from einops import rearrange

from src.data.dataprocessor.dataprocessor import ContoursProcessor


class Quantile(ContoursProcessor):
    """
    Class used for preprocessing and postprocessing
    and audio generation with DDSP.
    """

    def __init__(self, data_range: List[float], train_path: str,
                 ddsp_path: str) -> None:
        super().__init__(data_range, train_path, ddsp_path)

    def __read_from_pickle(self, path: str):
        """Read elements from a pickle file

        Args:
            path (str): path to the pickle file


        Yields:
            Iterator[np.ndarray]: element from the pickle
            file
        """

        with open(path, 'rb') as file:
            try:
                while True:
                    yield pickle.load(file)

            except EOFError:
                pass

    def fit_scalers(self):

        f0_scaler = QuantileTransformer(
            n_quantiles=self.n_quantiles,
            output_distribution=self.output_distribution)

        lo_scaler = QuantileTransformer(
            n_quantiles=self.n_quantiles,
            output_distribution=self.output_distribution)
        f0, lo = self.get_train_dataset()

        # clip contours to the violin range

        lo = np.clip(lo, self.lo_min, self.lo_max)
        f0 = np.clip(f0, self.f0_min, self.f0_max)

        # fit scaler
        f0_scaler = f0_scaler.fit(f0)
        lo_scaler = lo_scaler.fit(lo)

        # fit min and max post transform
        f0 = f0_scaler.transform(f0)
        lo = lo_scaler.transform(lo)

        self.f0_min_post, self.f0_max_post = f0.min(), f0.max()
        self.lo_min_post, self.lo_max_post = lo.min(), lo.max()

        return f0_scaler, lo_scaler

    def get_train_dataset(self):
        contours = next(self.__read_from_pickle(self.train_path))
        lo = contours["lo"].reshape(-1, 1)
        f0 = contours["f0"].reshape(-1, 1)
        return f0, lo

    def __rescale(self, x, o_m, o_M, n_m, n_M):
        return (n_M - n_m) / (o_M - o_m) * (x - o_M) + n_M

    def __set_ddsp(self, path: str) -> nn.Module:
        """Set pretrained ddsp model and freeze 
        its parameters

        Args:
            path (str): path to the ddsp model

        Returns:
            nn.Module: frozen ddsp model
        """
        if path is None:
            return None
        ddsp = torch.jit.load(path).cuda()
        # freeze ddsp:
        for p in ddsp.parameters():
            p.requires_grad = False
        return ddsp

    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """Pre processing, converting frequency contours from the 
        dataset, in the Hz scale to the specified range.

        Args:
            data (torch.Tensor): fundamental frequency contour (on
            MIDI scale) and loudness contour (dB scale) of shape
            (B L C) where C=2.

        Returns:
            torch.Tensor: scaled fundamental frequency and loudness contours
            concatenated on the L axis, tensor of shape (B (C*L))
        """

        # data of shape B L C
        f0, lo = rearrange(data.detach().cpu(), "b l c -> c (b l) 1")

        f0 = np.clip(f0, self.f0_min, self.f0_max)
        lo = np.clip(lo, self.lo_min, self.lo_max)

        # transform

        f0 = self.f0_scaler.transform(f0)
        lo = self.lo_scaler.transform(lo)

        # rescale to match [-1 1] range
        scaled_f0 = self.__rescale(f0, self.f0_min_post, self.f0_max_post, -1,
                                   1)
        scaled_lo = self.__rescale(lo, self.lo_min_post, self.lo_max_post, -1,
                                   1)

        out = rearrange([scaled_f0, scaled_lo],
                        "c (b l) 1 -> b (c l)",
                        l=self.sample_len)

        return torch.tensor(out).float().cuda()

    def inv_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply inverse transformation to the data tensor (rescaling)

        Args:
            data (torch.Tensor): contour of shape (B (C*L)) with C=2

        Returns:
            torch.Tensor: rescaled contour of shape (B (C*L)) with C=2
        """

        data = data.detach().cpu().numpy()
        f0, lo = rearrange(data, "b (c l) -> c (b l) 1", l=self.sample_len)

        # rescale f0, lo

        f0 = self.__rescale(f0, -1, 1, self.f0_min_post, self.f0_max_post)
        lo = self.__rescale(lo, -1, 1, self.lo_min_post, self.lo_max_post)

        f0 = self.f0_scaler.inverse_transform(f0)
        lo = self.lo_scaler.inverse_transform(lo)

        out = rearrange([f0, lo], " c (b l) 1 -> b (c l)", l=self.sample_len)

        return torch.tensor(out).float().cuda()

    def c2wav(self, data: torch.Tensor) -> torch.Tensor:
        """Contours to audio converting

        Args:
            data (torch.Tensor): contour of (f0, lo) of shape 
            (B (C*L)) with C=2.

        Returns:
            torch.Tensor: corresponding waveform of shape (B L*sr)
        """
        # we need loudness contour to generate audio
        f0, lo = rearrange(data, "b (c l) -> c b l 1", c=2)
        # rescale to the frequency range
        f0 = 440 * torch.pow(2, (f0 - 69) / 12)
        # artificialy add 2db to the loudness contours
        return self.ddsp(f0, lo + 1.5).squeeze(-1)

    def postprocess(self, data: torch.Tensor) -> torch.Tensor:
        """Post processing, going from generated contours to
        the corresponding audio (DDSP model as synthesizer)

        Args:
            data (torch.Tensor): generated contours (in contours range)
            of shape (B (C*L)) with C=2.

        Returns:
            torch.Tensor: generated audio (sr=16kHz) of shape (B L*sr)
        """
        data = self.inv_transform(data)
        audio = self.c2wav(data)

        return audio