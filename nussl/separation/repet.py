#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The original REpeating Pattern Extraction Technique (REPET).
"""

import numpy as np
import scipy.fftpack as scifft
import scipy.spatial.distance

import mask_separation_base
import masks
from ..core import constants


class Repet(mask_separation_base.MaskSeparationBase):
    """Implements the original REpeating Pattern Extraction Technique algorithm using the beat spectrum.

    REPET is a simple method for separating a repeating background from a non-repeating foreground in an
    audio mixture. It assumes a single repeating period over the whole signal duration, and finds that
    period based on finding a peak in the beat spectrum. The period can also be provided exactly, or you
    can give ``Repet`` a guess of the min and max period. Once it has a period, it "overlays" spectrogram
    sections of length ``period`` to create a median model (the background).

    References:
        * Zafar Rafii and Bryan Pardo. "Audio Separation System and Method," US20130064379 A1, US 13/612,413, March 14,
          2013

    See Also:
        http://music.eecs.northwestern.edu/research.php?project=repet
        :ref:`The REPET Demo Example <repet_demo>`
        :class:`separation.repet_sim.RepetSim`

    Parameters:
        input_audio_signal (:class:`audio_signal.AudioSignal`): The :class:`audio_signal.AudioSignal` object that
         REPET will be run on. This makes a copy of ``input_audio_signal``
        min_period (float, optional): minimum time to look for repeating period in terms of seconds.
        max_period (float, optional): maximum time to look for repeating period in terms of seconds.
        period (float, optional): exact time that the repeating period is (in seconds).
        high_pass_cutoff (float, optional): value (in Hz) for the high pass cutoff filter.
        do_mono (bool, optional): Flattens :class:`audio_signal.AudioSignal` to mono before running the 
        algorithm (does not effect the input :class:`audio_signal.AudioSignal` object).
        use_find_period_complex (bool, optional): Will use a more complex peak picker to find the repeating period.
        use_librosa_stft (bool, optional): Calls librosa's stft function instead of nussl's
        matlab_fidelity (bool, optional): If True, does repet with the same settings as the original MATLAB
                        implementation of REPET, warts and all. This will override ``use_librosa_stft`` and set
                        it to ``False``.

    Examples:
        

    Attributes:
        background (:class:`audio_signal.AudioSignal`): Calculated background. This is ``None`` until :func:`run()` is 
            called.
        foreground (:class:`audio_signal.AudioSignal`): Calculated foreground. This is ``None`` until 
            :func:`make_audio_signals()` is called.
        beat_spectrum (:obj:`np.array`): Beat spectrum calculated by Repet.
        use_find_period_complex (bool): Determines whether to use complex peak picker to find the repeating period.
        repeating_period (int): Repeating period in units of hops (stft time bins)
        stft (:obj:`np.ndarray`): Local copy of the STFT input from ``input_audio_array``
        mangitude_spectrogram (:obj:`np.ndarray`): Local copy of the magnitude spectrogram

    """
    def __init__(self, input_audio_signal, min_period=None, max_period=None, period=None, high_pass_cutoff=100.0,
                 do_mono=False, use_find_period_complex=False,
                 use_librosa_stft=constants.USE_LIBROSA_STFT, matlab_fidelity=False,
                 mask_type=mask_separation_base.MaskSeparationBase.SOFT_MASK, mask_threshold=0.5):
        super(Repet, self).__init__(input_audio_signal=input_audio_signal, mask_type=mask_type,
                                    mask_threshold=mask_threshold)

        # Check input parameters
        if (min_period or max_period) and period:
            raise ValueError('Cannot set both period and (min_period or max_period)!')

        self.high_pass_cutoff = float(high_pass_cutoff)
        self.background = None
        self.foreground = None
        self.beat_spectrum = None
        self.use_find_period_complex = use_find_period_complex
        self.use_librosa_stft = use_librosa_stft

        self.repeating_period = None
        self.magnitude_spectrogram = None
        self.stft = None
        self.matlab_fidelity = matlab_fidelity
        self._is_period_converted_to_hops = False

        if self.matlab_fidelity:
            self.use_librosa_stft = False

        # TODO: stereo doesn't do true stereo REPET (see TODO below)
        if do_mono:
            self.audio_signal.to_mono(overwrite=True)

        # Set period parameters
        self.min_period, self.max_period, self.period = None, None, None
        if period is None:
            self.min_period = 0.8 if min_period is None else min_period
            self.max_period = min(8, self.audio_signal.signal_duration / 3) if max_period is None else max_period
        else:
            self.period = period
            if not self._is_period_converted_to_hops:
                self.period = self._update_period(self.period)
                self._is_period_converted_to_hops = True

    def run(self):
        """ Runs the original REPET algorithm

        Returns:
            masks (:obj:`MaskBase`): A :obj:`MaskBase`-derived object with repeating background time-frequency data.
            (to get the corresponding non-repeating foreground run :func:`make_audio_signals`)

        Example:
            
        .. code-block:: python
            :linenos:
            
            signal = nussl.AudioSignal(path_to_input_file='input_name.wav')

            # Set up and run Repet
            repet = nussl.Repet(signal)  # Returns a soft mask by default
            masks = repet.run() # or repet()

            # Get audio signals
            background, foreground = repet.make_audio_signals()

            # output the background
            background.write_audio_to_file('background.wav')

        """
        # High pass filter cutoff freq. (in # of freq. bins), +1 to match MATLAB implementation
        self.high_pass_cutoff = int(np.ceil(self.high_pass_cutoff * (self.stft_params.n_fft_bins - 1) /
                                            self.audio_signal.sample_rate)) + 1

        # the MATLAB implementation had
        low = 1 if self.matlab_fidelity else 0

        self._compute_spectrograms()
        self.repeating_period = self._calculate_repeating_period()

        # separate the mixture background by masking
        background_stft = []
        background_mask = []
        for i in range(self.audio_signal.num_channels):
            repeating_mask = self._compute_repeating_mask(self.magnitude_spectrogram[:, :, i])

            repeating_mask[low:self.high_pass_cutoff, :] = 1  # high-pass filter the foreground
            background_mask.append(repeating_mask)

            # apply mask
            stft_with_mask = repeating_mask * self.stft[:, :, i]
            background_stft.append(stft_with_mask)

        # make a new audio signal for the background
        background_stft = np.array(background_stft).transpose((1, 2, 0))
        self._make_background_signal(background_stft)

        # make a mask and return
        background_mask = np.array(background_mask).transpose((1, 2, 0))
        background_mask = masks.SoftMask(background_mask)
        if self.mask_type == self.BINARY_MASK:
            background_mask = background_mask.mask_to_binary(self.mask_threshold)

        self.result_masks = [background_mask, background_mask.inverse_mask()]

        return self.result_masks

    def _compute_spectrograms(self):
        self.stft = self.audio_signal.stft(overwrite=True, remove_reflection=True, use_librosa=self.use_librosa_stft)
        self.magnitude_spectrogram = np.abs(self.stft)

    def get_beat_spectrum(self, recompute_stft=False):
        """Calculates and returns the beat spectrum for the audio signal associated with this object

        Args:
            recompute_stft (bool, Optional): Recompute the stft for the audio signal

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
        if recompute_stft or self.magnitude_spectrogram is None:
            self._compute_spectrograms()

        # TODO: Make this multi-channel. The np.mean() reduces the n channels to 1.
        self.beat_spectrum = self.compute_beat_spectrum(np.mean(np.square(self.magnitude_spectrogram),
                                                                axis=self.audio_signal._STFT_CHAN).T)
        return self.beat_spectrum

    def _calculate_repeating_period(self):
        # user provided a period, so no calculations to do
        if self.period is not None:
            return self.period

        # get beat spectrum
        self.beat_spectrum = self.get_beat_spectrum()

        if self.use_find_period_complex:
            self.repeating_period = self.find_repeating_period_complex(self.beat_spectrum)
        else:
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
        fft_power_spec = scifft.fft(power_spectrogram, axis=0)
        abs_fft = np.abs(fft_power_spec) ** 2
        autocorrelation_rows = np.real(scifft.ifft(abs_fft, axis=0)[:freq_bins, :])  # ifft over columns

        # normalization factor
        norm_factor = np.tile(np.arange(freq_bins, 0, -1), (time_bins, 1)).T
        autocorrelation_rows = autocorrelation_rows / norm_factor

        # compute the beat spectrum
        beat_spectrum = np.mean(autocorrelation_rows, axis=1)  # average over frequencies

        return beat_spectrum

    @staticmethod
    def find_repeating_period_simple(beat_spectrum, min_period, max_period):
        """Computes the repeating period of the sound signal using the beat spectrum.
           This algorithm just looks for the max value in the interval ``[min_period, max_period]``, inclusive.
           It discards the first value, and returns the period in units of stft time bins.

        Parameters:
            beat_spectrum (:obj:`np.array`): input beat spectrum array
            min_period (int): minimum possible period value
            max_period (int): maximum possible period value
            
        Returns:
             period (int): The period of the sound signal in stft time bins
             
        See Also:
            :func:`find_repeating_period_complex`
            
        """
        min_period, max_period = int(min_period), int(max_period)
        beat_spectrum = beat_spectrum[1:]  # discard the first element of beat_spectrum (lag 0)
        beat_spectrum = beat_spectrum[min_period - 1: max_period]

        if len(beat_spectrum) == 0:
            raise RuntimeError('min_period is larger than the beat spectrum!')

        period = np.argmax(beat_spectrum) + min_period

        return period

    @staticmethod
    def find_repeating_period_complex(beat_spectrum):
        """ A more complicated approach to finding the repeating period. Use this by setting 
        :attr:`use_find_period_complex`
        
        Args:
            beat_spectrum (:obj:`np.array`): input beat spectrum array

        Returns:
            period (int): The period of the sound signal in stft time bins
            
        See Also:
            :func:`find_repeating_period_simple`
        
        """
        auto_cosine = np.zeros((len(beat_spectrum), 1))

        for i in range(0, len(beat_spectrum) - 1):
            auto_cosine[i] = 1 - scipy.spatial.distance.cosine(beat_spectrum[0:len(beat_spectrum) - i],
                                                               beat_spectrum[i:len(beat_spectrum)])

        ac = auto_cosine[0:np.floor(auto_cosine.shape[0])/2]
        auto_cosine = np.vstack([ac[1], ac, ac[-2]])
        auto_cosine_diff = np.ediff1d(auto_cosine)
        sign_changes = auto_cosine_diff[0:-1]*auto_cosine_diff[1:]
        sign_changes = np.where(sign_changes < 0)[0]

        extrema_values = ac[sign_changes]

        e1 = np.insert(extrema_values, 0, extrema_values[0])
        e2 = np.insert(extrema_values, -1, extrema_values[-1])

        extrema_neighbors = np.stack((e1[0:-1], e2[1:]))

        m = np.amax(extrema_neighbors, axis=0)
        extrema_values = extrema_values.flatten()
        maxima = np.where(extrema_values >= m)[0]
        maxima = zip(sign_changes[maxima], extrema_values[maxima])
        maxima = maxima[1:]
        maxima = sorted(maxima, key=lambda x: -x[1])
        period = maxima[0][0]

        return period

    def _compute_repeating_mask(self, magnitude_spectrogram_channel):
        """Computes the soft mask for the repeating part using the magnitude spectrogram and the repeating period

        Parameters:
            magnitude_spectrogram_channel (:obj:`np.array`): 2D matrix containing the magnitude spectrogram of a signal
            
        Returns:
            (:obj:`np.array`): 2D matrix (Lf by Lt) containing the soft mask for the repeating part, elements of M 
            take on values in ``[0, 1]``

        """
        # this +1 is a kluge to make this implementation match the original MATLAB implementation
        period = self.repeating_period + 1
        freq_bins, time_bins = magnitude_spectrogram_channel.shape
        n_repetitions = int(np.ceil(float(time_bins) / period))
        one_period = freq_bins * period

        # Pad to make an integer number of repetitions. Pad with 'nan's to not affect the median.
        remainder = (period * n_repetitions) % time_bins
        mask_reshaped = np.hstack([magnitude_spectrogram_channel, float('nan') * np.zeros((freq_bins, remainder))])

        # reshape to take the median of each period
        mask_reshaped = np.reshape(mask_reshaped.T, (n_repetitions, one_period))

        # take median of repeating periods before and after the padding
        median_mask = np.nanmedian(mask_reshaped, axis=0)

        # reshape to it's original shape
        median_mask = np.reshape(np.tile(median_mask, (n_repetitions, 1)), (n_repetitions * period, freq_bins)).T
        median_mask = median_mask[:, :time_bins]

        # take minimum of computed mask and original input and scale
        min_median_mask = np.minimum(median_mask, magnitude_spectrogram_channel)
        mask = (min_median_mask + constants.EPSILON) / (magnitude_spectrogram_channel + constants.EPSILON)

        return mask

    def update_periods(self):
        """ Will update periods for use with :func:`find_repeating_period_simple`.

        Updates from seconds to stft time bin values.
        Call this if you haven't done :func:`run()` or else you won't get good results.

        Example:
            
        .. code-block:: python
            :linenos:

            a = nussl.AudioSignal('path/to/file.wav')
            r = nussl.Repet(a)

            beat_spectrum = r.get_beat_spectrum()
            r.update_periods()
            repeating_period = r.find_repeating_period_simple(beat_spectrum, r.min_period, r.max_period)

        """
        if self._is_period_converted_to_hops:
            self.period = self._update_period(self.period) if self.period is not None else None
            self.min_period = self._update_period(self.min_period) if self.min_period is not None else None
            self.max_period = self._update_period(self.max_period) if self.max_period is not None else None
            self._is_period_converted_to_hops = True

    def _update_period(self, period):
        period = float(period)
        result = period * self.audio_signal.sample_rate
        result += self.stft_params.window_length / self.stft_params.window_overlap - 1
        result /= self.stft_params.window_overlap
        return int(np.ceil(result))

    def _make_background_signal(self, background_stft):
        self.background = self.audio_signal.make_copy_with_stft_data(background_stft, verbose=False)
        self.background.istft(self.stft_params.window_length, self.stft_params.hop_length, self.stft_params.window_type,
                              overwrite=True, use_librosa=self.use_librosa_stft,
                              truncate_to_length=self.audio_signal.signal_length)

    def plot(self, output_file, **kwargs):
        """
        Creates a plot of the beat spectrum and outputs to output_file.

        Parameters:
            output_file (string) : string representing a path to the desired output file to be created.
            title: (string) Title to put on the plot
            show_repeating_period: (bool) if True, then adds a vertical line where repet things
                                the repeating period is (if the repeating period has been computed already)

        Example:
        
        .. code-block:: python
            :linenos:

            signal = nussl.AudioSignal('Sample.wav')
            repet = nussl.Repet(signal)

            repet.plot('new_beat_spec_plot.png', title="Beat Spectrum of Sample.wav", show_repeating_period=True)
        """
        import matplotlib.pyplot as plt
        plt.close('all')
        title = None
        show_repeating_period = False

        if len(kwargs) != 0:
            if 'title' in kwargs:
                title = kwargs['title']
            if 'show_repeating_period' in kwargs:
                show_repeating_period = kwargs['show_repeating_period']

        beat_spec = self.get_beat_spectrum()
        time_vect = np.linspace(0.0, self.audio_signal.signal_duration, num=len(beat_spec))
        plt.plot(time_vect, beat_spec)

        if self.repeating_period is not None and show_repeating_period:
            stft_vector = np.linspace(0.0, self.audio_signal.signal_duration, self.audio_signal.stft_length)
            rep = stft_vector[self.repeating_period]
            plt.plot((rep, rep), (0, np.max(beat_spec)), 'g--', label='Repeating period')
            # plt.plot((self.repeating_period, self.repeating_period), (-1e20, 1e20), 'g--')
            plt.ylim((0.0, np.max(beat_spec) * 1.1))

        title = title if title is not None else 'Beat Spectrum for {}'.format(self.audio_signal.file_name)
        plt.title(title)

        plt.xlabel('Time (s)')
        plt.ylabel('Beat Strength')
        plt.grid('on')

        plt.axis('tight')
        plt.savefig(output_file)

    def make_audio_signals(self):
        """ Returns the background and foreground audio signals. You must have run :func:`run()` prior
        to calling this function. This function will return ``None`` if :func:`run()` has not been called.
        
        Order of the list is ``[self.background, self.foreground]`` 

        Returns:
            (list): List containing two :class:`audio_signal.AudioSignal` objects, one for the calculated background
            and the next for the remaining foreground, in that order.

        Example:
            
        .. code-block:: python
            :linenos:
            
            # set up AudioSignal object
            signal = nussl.AudioSignal('path_to_file.wav')

            # set up and run repet
            repet = nussl.Repet(signal)
            repet.run()

            # get audio signals (AudioSignal objects)
            background, foreground = repet.make_audio_signals()
            
        """
        if self.background is None:
            raise ValueError('Cannot make audio signals prior to running algorithm!')

        foreground_array = self.audio_signal.audio_data - self.background.audio_data
        self.foreground = self.audio_signal.make_copy_with_audio_data(foreground_array)
        return [self.background, self.foreground]
