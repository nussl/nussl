import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter, uniform_filter

from .. import MaskSeparationBase, SeparationException
from ..benchmark import HighLowPassFilter


class FT2D(MaskSeparationBase):
    """
    This separation method is based on using 2DFT image processing for source
    separation [1].

    The algorithm has five main steps:

    1. Take the 2DFT of Magnitude STFT.
    2. Identify peaks in the 2DFT (these correspond to repeating patterns).
    3. Mask everything but the peaks and invert the masked 2DFT back to a magnitude STFT.
    4. Take the residual peakless 2DFT and invert that to a magnitude STFT.
    5. Compare the two magnitude STFTs to construct masks for the foreground and
       the background.

    The algorithm runs on each channel independently. The masks can be constructed
    in two ways: either using the background (peaky) 2DFT as the reference for making 
    masks, or using the foreground (peakless) 2DFT as the reference for making masks.
    This behavior can be toggled via `use_bg_2dft=[True, False]`. Using the background
    biases the algorithm towards separating repeating patterns whereas using
    the foreground biases it towards preserving micromodulation in the foreground.

    There are two ways to identify peaks: either using the `original` method, as
    laid out in the paper which identifies peaks in a binary way as being above
    some threshold, or by using the `local_std` way, which was developed and investigated
    in Chapter 2 in [2]. This method identifies peaks in a soft way and can thus
    be used to construct soft masks. It generally performs better.

    The main hyperparameter to consider is the `neighborhood_size`, which determines
    how big of a filter to apply when looking for peaks. It is two dimensional, but
    generally keeping 1D horizontal filters is the way to go.

    References:

        [1] Seetharaman, Prem, Fatemeh Pishdadian, and Bryan Pardo.
            "Music/Voice Separation Using the 2D Fourier Transform." 
            2017 IEEE Workshop on Applications of Signal Processing to 
            Audio and Acoustics (WASPAA). IEEE, 2017.

        [2] Seetharaman, Prem. Bootstrapping the Learning Process for Computer Audition. 
            Diss. Northwestern University, 2019.

    Args:
        input_audio_signal (AudioSignal): Signal to separate.

        neighborhood_size (tuple, optional): 2-tuple of ints telling the filter size to look for
          peaks in frequency and time: `(f, t)`. Defaults to (1, 25).

        high_pass_cutoff (float, optional): Time-frequency bins below this cutoff will be
          assigned to the background. Defaults to 100.0.

        quadrants_to_keep (tuple, optional): Each quadrant of the 2DFT can be filtered out
          when separating. Can be used for separating out upward spectro-temporal patterns
          (via `(0, 2)` from downward ones `(1, 3)`. Defaults to (0,1,2,3).

        filter_approach (str, optional): One of 'original' or 'local_std'. Which 
          filtering approach to apply to identify peaks. Defaults to 'local_std'.

        use_bg_2dft (bool, optional): Whether to use the background or foreground 2DFT
          as the reference for constructing masks. Defaults to True.

        mask_type (str, optional): Mask type. Defaults to 'soft'.

        mask_threshold (float, optional): Masking threshold. Defaults to 0.5.

    """

    def __init__(self, input_audio_signal, neighborhood_size=(1, 25), high_pass_cutoff=100.0,
                 quadrants_to_keep=(0, 1, 2, 3), filter_approach='local_std', use_bg_2dft=True,
                 mask_type='soft', mask_threshold=0.5):

        super().__init__(
            input_audio_signal=input_audio_signal,
            mask_type=mask_type,
            mask_threshold=mask_threshold
        )

        self.neighborhood_size = neighborhood_size
        self.result_masks = None
        self.quadrants_to_keep = quadrants_to_keep
        self.use_bg_2dft = use_bg_2dft
        self.high_pass_cutoff = high_pass_cutoff
        self.bg_ft2d = None
        self.fg_ft2d = None

        if filter_approach == 'original':
            self.filter_func = self.filter_local_maxima
        elif filter_approach == 'local_std':
            self.filter_func = self.filter_local_maxima_with_std
        else:
            allowed_filter_approaches = ['original', 'local_std']
            raise SeparationException(
                f'filter approach must be one of {allowed_filter_approaches}')

    def run(self):
        high_low = HighLowPassFilter(self.audio_signal, self.high_pass_cutoff)
        high_pass_masks = high_low.run()

        background_masks = []
        foreground_masks = []

        for ch in range(self.audio_signal.num_channels):
            background_mask, foreground_mask = self.compute_ft2d_mask(
                self.ft2d, ch
            )
            background_masks.append(background_mask)
            foreground_masks.append(foreground_mask)

        background_masks = np.stack(background_masks, axis=-1)
        foreground_masks = np.stack(foreground_masks, axis=-1)

        _masks = np.stack([background_masks, foreground_masks], axis=-1)

        self.result_masks = []

        for i in range(_masks.shape[-1]):
            mask_data = _masks[..., i]
            if self.mask_type == self.MASKS['binary']:
                mask_data = _masks[..., i] == np.max(_masks, axis=-1)

            if i == 0:
                mask_data = np.maximum(mask_data, high_pass_masks[i].mask)
            elif i == 1:
                mask_data = np.minimum(mask_data, high_pass_masks[i].mask)

            mask = self.mask_type(mask_data)
            self.result_masks.append(mask)

        return self.result_masks

    def _preprocess_audio_signal(self):
        super()._preprocess_audio_signal()

        self.ft2d = np.stack([
            np.fft.fft2(np.abs(self.stft[:, :, i]))
            for i in range(self.audio_signal.num_channels)],
            axis=-1
        )

    @staticmethod
    def filter_quadrants(data, quadrants_to_keep):
        """
        Filters the quadrants of the incoming 2DFT, deleting the ones not in 
        `quadrants_to_keep`. Those are zero'd out. This can be used to separate
        upward moving from downward moving spectro-temporal patterns.
        
        .. code-block:: none
        
            1: shape[0] // 2:, :shape[1] // 2
            2: :shape[0] // 2, :shape[1] // 2
            3: :shape[0] // 2, shape[1] // 2:
            4: shape[0] // 2:, shape[1] // 2:
        
        Args:
            data ([type]): [description]
            quadrants_to_keep ([type]): [description]
        
        Returns:
            [type]: [description]
        """

        shape = data.shape
        for quadrant in range(4):
            if quadrant not in quadrants_to_keep:
                if quadrant == 0:
                    data[shape[0] // 2:, :shape[1] // 2] = 0
                elif quadrant == 1:
                    data[:shape[0] // 2, :shape[1] // 2] = 0
                elif quadrant == 2:
                    data[:shape[0] // 2, shape[1] // 2:] = 0
                elif quadrant == 3:
                    data[shape[0] // 2:, shape[1] // 2:] = 0
        return data

    def compute_ft2d_mask(self, ft2d, ch):
        bg_ft2d, fg_ft2d = self.filter_func(ft2d[:, :, ch])

        self.bg_ft2d = self.filter_quadrants(bg_ft2d, self.quadrants_to_keep)
        self.fg_ft2d = self.filter_quadrants(fg_ft2d, self.quadrants_to_keep)

        _stft = np.abs(self.stft)[:, :, ch] + 1e-7
        _stft = _stft

        if self.use_bg_2dft:
            ft2d_used = self.bg_ft2d
        else:
            ft2d_used = self.fg_ft2d

        est_stft = np.minimum(np.abs(np.fft.ifft2(ft2d_used)), _stft)
        est_mask = est_stft / _stft
        est_mask /= (est_mask + 1e-7).max()

        if self.use_bg_2dft:
            bg_mask = est_mask
            fg_mask = 1 - bg_mask
        else:
            fg_mask = est_mask
            bg_mask = 1 - fg_mask

        return bg_mask, fg_mask

    def filter_local_maxima_with_std(self, ft2d):
        data = np.abs(np.fft.fftshift(ft2d))
        data /= (np.max(data) + 1e-7)

        data_max = maximum_filter(data, self.neighborhood_size)
        data_mean = uniform_filter(data, self.neighborhood_size)
        data_mean_sq = uniform_filter(data ** 2, self.neighborhood_size)
        data_std = np.sqrt(data_mean_sq - data_mean ** 2) + 1e-7

        maxima = ((data_max - data_mean) / data_std)
        fraction_of_local_max = (data == data_max)
        maxima *= fraction_of_local_max
        maxima = maxima.astype(float)
        maxima /= (np.max(maxima) + 1e-7)

        maxima = np.maximum(maxima, np.fliplr(maxima), np.flipud(maxima))
        maxima = np.fft.ifftshift(maxima)

        background_ft2d = np.multiply(maxima, ft2d)
        foreground_ft2d = np.multiply(1 - maxima, ft2d)

        return background_ft2d, foreground_ft2d

    def filter_local_maxima(self, ft2d):
        data = np.abs(np.fft.fftshift(ft2d))
        data /= (np.max(data) + 1e-7)
        threshold = np.std(data)

        data_max = maximum_filter(data, self.neighborhood_size)
        maxima = (data == data_max)
        data_min = minimum_filter(data, self.neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        maxima = np.maximum(maxima, np.fliplr(maxima), np.flipud(maxima))
        maxima = np.fft.ifftshift(maxima)

        background_ft2d = np.multiply(maxima, ft2d)
        foreground_ft2d = np.multiply(1 - maxima, ft2d)

        return background_ft2d, foreground_ft2d
