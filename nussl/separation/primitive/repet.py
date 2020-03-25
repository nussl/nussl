import numpy as np
import scipy.fftpack as scifft

from .. import MaskSeparationBase, SeparationException
from ..benchmark import HighLowPassFilter
from ...core import constants


class Repet(MaskSeparationBase):
    """Implements the original REpeating Pattern Extraction Technique algorithm 
    using the beat spectrum.

    REPET is a simple method for separating a repeating background from a 
    non-repeating foreground in an audio mixture. It assumes a single repeating 
    period over the whole signal duration, and finds that period based on finding 
    a peak in the beat spectrum. The period can also be provided exactly, or you
    can give ``Repet`` a guess of the min and max period. Once it has a period, 
    it "overlays" spectrogram sections of length ``period`` to create a median 
    model (the background).
    
    References:

    [1] Rafii, Zafar, and Bryan Pardo.
        "Repeating pattern extraction technique (REPET): A simple method for 
        music/voice separation." IEEE transactions on audio, speech, 
        and language processing 21.1 (2012): 73-84.

    Args:
        input_audio_signal (AudioSignal): Signal to separate.

        min_period (float, optional): minimum time to look for repeating period in 
          terms of seconds.
        
        max_period (float, optional): maximum time to look for repeating period in 
          terms of seconds.
        
        period (float, optional): exact time that the repeating period is 
          (in seconds).
        
        high_pass_cutoff (float, optional): value (in Hz) for the high pass 
          cutoff filter.
                        
        mask_type (str, optional): Mask type. Defaults to 'soft'.

        mask_threshold (float, optional): Masking threshold. Defaults to 0.5.

    """

    def __init__(self, input_audio_signal, min_period=None, max_period=None,
                 period=None, high_pass_cutoff=100.0, mask_type='soft',
                 mask_threshold=0.5):

        super().__init__(
            input_audio_signal=input_audio_signal,
            mask_type=mask_type,
            mask_threshold=mask_threshold)

        # Check input parameters
        if (min_period or max_period) and period:
            raise SeparationException(
                'Cannot set both period and (min_period or max_period)!')

        self._is_period_converted_to_hops = False
        self.period = period

        if self.period is None:
            self.min_period = min_period if min_period else .8
            max_period = max_period if max_period else 8
            self.max_period = min(max_period, self.audio_signal.signal_duration / 3)
        else:
            self.period = self._update_period(period)
            self._is_period_converted_to_hops = True

        self.high_pass_cutoff = high_pass_cutoff
        self.magnitude_spectrogram = None
        self.repeating_period = None
        self.beat_spectrum = None

    def run(self):
        high_low = HighLowPassFilter(self.audio_signal, self.high_pass_cutoff)
        high_pass_masks = high_low.run()

        self.magnitude_spectrogram = np.abs(self.stft)

        background_masks = []
        foreground_masks = []

        self.repeating_period = self._calculate_repeating_period()

        for ch in range(self.audio_signal.num_channels):
            background_mask = self._compute_repeating_mask(
                self.magnitude_spectrogram[..., ch])
            foreground_mask = 1 - background_mask

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

    def get_beat_spectrum(self):
        """
        Calculates and returns the beat spectrum for the audio signal associated 
        with this object

        Returns:
            beat_spectrum (np.array): beat spectrum for the audio file

        Example:

        .. code-block:: python
            :linenos:
            
            # Set up audio signal
            signal = nussl.AudioSignal('path_to_file.wav')

            # Set up a Repet object
            repet = nussl.Repet(signal)

            # I don't have to run repet to get a beat spectrum for signal
            beat_spec = repet.get_beat_spectrum()
            
        """
        # TODO: Make this multi-channel. The np.mean() reduces the n channels to 1.       
        self.beat_spectrum = self.compute_beat_spectrum(
            np.mean(np.square(self.magnitude_spectrogram),
                    axis=constants.STFT_CHAN_INDEX).T
        )
        return self.beat_spectrum

    def _calculate_repeating_period(self):
        # user provided a period, so no calculations to do
        if self.period is not None:
            return self.period

        # get beat spectrum
        self.beat_spectrum = self.get_beat_spectrum()

        # update the min and max so they're in units of time bin indices
        if not self._is_period_converted_to_hops:
            self.min_period = self._update_period(self.min_period)
            self.max_period = self._update_period(self.max_period)
            self._is_period_converted_to_hops = True

        self.repeating_period = self.find_repeating_period_simple(self.beat_spectrum,
                                                                  self.min_period, self.max_period)

        return self.repeating_period

    @staticmethod
    def compute_beat_spectrum(power_spectrogram):
        """ Computes the beat spectrum averages (over freq's) the autocorrelation matrix of a one-sided spectrogram.

        The autocorrelation matrix is computed by taking the autocorrelation of each row of the spectrogram and
        dismissing the symmetric half.

        Args:
            power_spectrogram (:obj:`np.array`): 2D matrix containing the one-sided power spectrogram of an audio signal
            
        Returns:
            (:obj:`np.array`): array containing the beat spectrum based on the power spectrogram
            
        See Also:
            J Foote's original derivation of the Beat Spectrum: 
            Foote, Jonathan, and Shingo Uchihashi. "The beat spectrum: A new approach to rhythm analysis." 
            Multimedia and Expo, 2001. ICME 2001. IEEE International Conference on. IEEE, 2001.
            (`See PDF here <http://rotorbrain.com/foote/papers/icme2001.pdf>`_)
            
        """
        freq_bins, time_bins = power_spectrogram.shape

        # row-wise autocorrelation according to the Wiener-Khinchin theorem
        power_spectrogram = np.vstack([power_spectrogram, np.zeros_like(power_spectrogram)])

        nearest_power_of_two = 2 ** np.ceil(np.log(power_spectrogram.shape[0]) / np.log(2))
        pad_amount = int(nearest_power_of_two - power_spectrogram.shape[0])
        power_spectrogram = np.pad(power_spectrogram, ((0, pad_amount), (0, 0)), 'constant')
        fft_power_spec = scifft.fft(power_spectrogram, axis=0)
        abs_fft = np.abs(fft_power_spec) ** 2
        autocorrelation_rows = np.real(
            scifft.ifft(abs_fft, axis=0)[:freq_bins, :])  # ifft over columns

        # normalization factor
        norm_factor = np.tile(np.arange(freq_bins, 0, -1), (time_bins, 1)).T
        autocorrelation_rows = autocorrelation_rows / norm_factor

        # compute the beat spectrum
        beat_spectrum = np.mean(autocorrelation_rows, axis=1)
        # average over frequencies

        return beat_spectrum

    @staticmethod
    def find_repeating_period_simple(beat_spectrum, min_period, max_period):
        """
        Computes the repeating period of the sound signal using the beat spectrum.
        This algorithm just looks for the max value in the interval 
        ``[min_period, max_period]``, inclusive. It discards the first value, and 
        returns the period in units of stft time bins.

        Args:
            beat_spectrum (:obj:`np.array`): input beat spectrum array
            min_period (int): minimum possible period value
            max_period (int): maximum possible period value
            
        Returns:
             period (int): The period of the sound signal in stft time bins
             
        See Also:
            :func:`find_repeating_period_complex`
            
        """
        min_period, max_period = int(min_period), int(max_period)
        # discard the first element of beat_spectrum (lag 0)
        beat_spectrum = beat_spectrum[1:]
        beat_spectrum = beat_spectrum[min_period - 1: max_period]

        if len(beat_spectrum) == 0:
            raise SeparationException('min_period is larger than the beat spectrum!')

        period = np.argmax(beat_spectrum) + min_period

        return period

    def _compute_repeating_mask(self, magnitude_spectrogram_channel):
        """
        Computes the soft mask for the repeating part using the magnitude 
        spectrogram and the repeating period

        Args:
            magnitude_spectrogram_channel (:obj:`np.array`): 2D matrix containing the 
              magnitude spectrogram of a signal
            
        Returns:
            (:obj:`np.array`): 2D matrix (Lf by Lt) containing the soft mask for the 
              repeating part, elements of M take on values in ``[0, 1]``

        """
        period = self.repeating_period
        freq_bins, time_bins = magnitude_spectrogram_channel.shape
        n_repetitions = int(np.ceil(float(time_bins) / period))
        one_period = freq_bins * period

        # Pad to make an integer number of repetitions. Pad with 'nan's to not affect the median.
        remainder = (period * n_repetitions) % time_bins
        mask_reshaped = np.hstack(
            [magnitude_spectrogram_channel,
             float('nan') * np.zeros((freq_bins, remainder))]
        )

        # reshape to take the median of each period
        mask_reshaped = np.reshape(mask_reshaped.T, (n_repetitions, one_period))

        # take median of repeating periods before and after the padding
        median_mask = np.nanmedian(mask_reshaped, axis=0)

        # reshape to it's original shape
        median_mask = np.reshape(
            np.tile(median_mask, (n_repetitions, 1)),
            (n_repetitions * period, freq_bins)).T
        median_mask = median_mask[:, :time_bins]

        # take minimum of computed mask and original input and scale
        min_median_mask = np.minimum(median_mask, magnitude_spectrogram_channel)
        mask = (min_median_mask + constants.EPSILON) / (
                    magnitude_spectrogram_channel + constants.EPSILON)

        return mask

    def _update_period(self, period):
        period = float(period)
        secs_per_bin = self.stft_params.hop_length / self.audio_signal.sample_rate
        bins_in_period = period / secs_per_bin
        period_in_frames = int(np.ceil(bins_in_period))
        return period_in_frames
