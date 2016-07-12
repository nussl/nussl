#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.fftpack as scifft
import scipy.spatial.distance

import spectral_utils
import separation_base
import constants
from audio_signal import AudioSignal


class Repet(separation_base.SeparationBase):
    """Implements the original REpeating Pattern Extraction Technique algorithm using the beat spectrum.

    REPET is a simple method for separating the repeating background from the non-repeating foreground in a piece of
    audio mixture.

    References:

        * Zafar Rafii and Bryan Pardo. "Audio Separation System and Method," US20130064379 A1, US 13/612,413, March 14,
          2013

    See Also:
        http://music.eecs.northwestern.edu/research.php?project=repet

    Parameters:
        input_audio_signal: (AudioSignal object) The AudioSignal object that has the
                            audio data that REPET will be run on.
        min_period: (Optional) (float) minimum time to look for repeating period in terms of seconds.
        max_period: (Optional) (float) maximum time to look for repeating period in terms of seconds.
        period: (Optional) (float) exact time that the repeating period is (in seconds).
        high_pass_cutoff: (Optional) (float) value (in Hz) for the high pass cutoff filter.

    Examples:
        :ref:`The REPET Demo Example <repet_demo>`
    """
    def __init__(self, input_audio_signal, min_period=None, max_period=None, period=None, high_pass_cutoff=None,
                 do_mono=False, use_find_period_complex=False):
        self.__dict__.update(locals())
        super(Repet, self).__init__(input_audio_signal=input_audio_signal)
        self.high_pass_cutoff = 100.0 if high_pass_cutoff is None else float(high_pass_cutoff)
        self.bkgd = None
        self.fgnd = None
        self.beat_spectrum = None
        self.use_find_period_complex = use_find_period_complex

        self.repeating_period = None
        self.magnitude_spectrogram = None
        self.stft = None
        self.repeating_period = None

        #TODO: stereo doesn't do true stereo REPET (see TODO below)
        if do_mono:
            self.audio_signal.to_mono(overwrite=True)

        if (min_period or max_period) and period:
            raise ValueError('Cannot set both period and (min_period or max_period)!')

        # Set period parameters
        self.min_period, self.max_period, self.period = None, None, None
        if period is None:
            self.min_period = 0.8 if min_period is None else min_period
            self.max_period = min(8, self.audio_signal.signal_duration / 3) if max_period is None else max_period
        else:
            self.period = period
            self.period = self._update_period(self.period)

    def run(self):
        """Runs the REPET algorithm

        Returns:
            y (AudioSignal): repeating background (M by N) containing M channels and N time samples
            (the corresponding non-repeating foreground is equal to x-y)

        Example:
             ::
            signal = nussl.AudioSignal(path_to_input_file='input_name.wav')

            # Set up window parameters
            signal.stft_params.window_length = 2048
            signal.stft_params.window_type = nussl.WindowType.HAMMING

            # Set up and run Repet
            repet = nussl.Repet(signal)
            repet.run()

        """
        # High pass filter cutoff freq. (in # of freq. bins), +1 to match MATLAB implementation
        self.high_pass_cutoff = int(np.ceil(self.high_pass_cutoff * (self.stft_params.n_fft_bins - 1) /
                                            self.audio_signal.sample_rate)) + 1

        self._compute_spectrum()
        self.repeating_period = self._calculate_repeating_period()

        # separate the mixture background by masking
        bkgd = np.zeros_like(self.audio_signal.audio_data)
        for i in range(self.audio_signal.num_channels):
            repeating_mask = self._compute_repeating_mask(self.magnitude_spectrogram[:, :, i])
            repeating_mask[1:self.high_pass_cutoff, :] = 1  # high-pass filter the foreground

            # apply mask
            stft_with_mask = repeating_mask * self.stft[:, :, i]

            y = spectral_utils.e_istft(stft_with_mask, self.stft_params.window_length,
                                       self.stft_params.hop_length, self.stft_params.window_type,
                                       reconstruct_reflection=True, remove_padding=False)

            bkgd[i, ] = y[:self.audio_signal.signal_length]

        self.bkgd = AudioSignal(audio_data_array=bkgd, sample_rate=self.audio_signal.sample_rate)

        return self.bkgd

    def _compute_spectrum(self):
        self.stft = self.audio_signal.stft(overwrite=True, remove_reflection=True, use_librosa=False)
        self.magnitude_spectrogram = np.abs(self.stft)

    def get_beat_spectrum(self, recompute_stft=False):
        """Calculates and returns the beat spectrum for the audio signal associated with this object

        Args:
            recompute_stft: (Optional) (bool) Recompute the stft for the audio signal

        Returns:
            beat_spectrum (np.array): beat spectrum for the audio file

        EXAMPLE::

            # Set up audio signal
            signal = nussl.AudioSignal('path_to_file.wav')

            # Set up a Repet object
            repet = nussl.Repet(signal)

            # I don't have to run repet to get a beat spectrum for signal
            beat_spec = repet.get_beat_spectrum()
        """
        if recompute_stft or self.magnitude_spectrogram is None:
            self._compute_spectrum()

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
            # update the min and max so they're in units of frequency bin indices
            self.min_period = self._update_period(self.min_period)
            self.max_period = self._update_period(self.max_period)
            self.repeating_period = self.find_repeating_period_simple(self.beat_spectrum,
                                                                      self.min_period, self.max_period)
        return self.repeating_period

    @staticmethod
    def compute_beat_spectrum(power_spectrum):
        """Computes the beat spectrum averages (over freq's) the autocorrelation matrix of a one-sided spectrogram.

         The autocorrelation matrix is computed by taking the autocorrelation of each row of the spectrogram and
         dismissing the symmetric half.

        Parameters:
            power_spectrum (np.array): 2D matrix containing the one-sided power
            spectrogram of the audio signal (Lf by Lt by num channels)
        Returns:
            b (np.array): array containing the beat spectrum based on the power spectrogram
        """
        freq_bins, time_bins = power_spectrum.shape

        # row-wise autocorrelation according to the Wiener-Khinchin theorem
        power_spectrum = np.vstack([power_spectrum, np.zeros_like(power_spectrum)])
        fft_power_spec = scifft.fft(power_spectrum, axis=0)
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
           This algorithm just looks for the max value in the interval [min_period, max_period] inclusive.
           It discards the first value, and returns the period in units of stft time bins.

        Parameters:
            beat_spectrum (np.array): input beat spectrum array
            min_period (int): minimum possible period value
            max_period (int): maximum possible period value
        Returns:
             period (int) : The period of the sound signal in stft time bins
        """
        min_period, max_period = int(min_period), int(max_period)
        beat_spectrum = beat_spectrum[1:]  # discard the first element of beat_spectrum (lag 0)
        beat_spectrum = beat_spectrum[min_period - 1: max_period]
        period = np.argmax(beat_spectrum) + min_period

        return period

    @staticmethod
    def find_repeating_period_complex(beat_spectrum):
        """

        Args:
            beat_spectrum:

        Returns:

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
            magnitude_spectrogram_channel (np.array): 2D matrix containing the magnitude spectrogram of a signal
        Returns:
            M (np.array): 2D matrix (Lf by Lt) containing the soft mask for the repeating part, elements of M take on
            values in [0,1]

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
        split_point = freq_bins * (time_bins - (n_repetitions - 1) * period)
        median1 = np.median(mask_reshaped[0:n_repetitions, 0:split_point], axis=0)  # before padding
        median2 = np.median(mask_reshaped[0:n_repetitions-1, split_point:one_period], axis=0)  # after padding
        mask_reshaped = np.hstack([median1, median2])  # put them back together

        # reshape to it's original shape
        mask_reshaped = np.reshape(np.tile(mask_reshaped, (n_repetitions, 1)), (n_repetitions * period, freq_bins)).T
        mask_reshaped = mask_reshaped[:, 0:time_bins]

        # take minimum of computed mask and original input and scale
        mask_reshaped = np.minimum(mask_reshaped, magnitude_spectrogram_channel)
        mask = (mask_reshaped + constants.EPSILON) / (magnitude_spectrogram_channel + constants.EPSILON)

        return mask

    def update_periods(self):
        """
        Will update periods for use with self.find_repeating_period_simple(). Updates from seconds to stft bin values.
        Call this if you haven't done self.run() or else you won't get good results
        Examples:
            ::
            a = nussl.AudioSignal('path/to/file.wav')
            r = nussl.Repet(a)

            beat_spectrum = r.get_beat_spectrum()
            r.update_periods()
            repeating_period = r.find_repeating_period_simple(beat_spectrum, r.min_period, r.max_period)

        """
        self.period = self._update_period(self.period) if self.period is not None else None
        self.min_period = self._update_period(self.min_period) if self.min_period is not None else None
        self.max_period = self._update_period(self.max_period) if self.max_period is not None else None

    def _update_period(self, period):
        period = float(period)
        result = period * self.audio_signal.sample_rate
        result += self.stft_params.window_length / self.stft_params.window_overlap - 1
        result /= self.stft_params.window_overlap
        return int(np.ceil(result))

    def plot(self, output_file, **kwargs):
        """ Creates a plot of either the beat spectrum or similarity matrix for this file and outputs to output_file.
            By default, the repet_type is used to determine which to plot,
            (original -> beat spectrum. sim -> similarity matrix)
            You can override this by passing in plot_beat_spectrum=True or plot_sim_matrix=True as parameters.
            You cannot set both of these overrides simultaneously.

        Parameters:
            output_file (string) : string representing a path to the desired output file to be created.
            plot_beat_spectrum (Optional[bool]) : Setting this will force plotting the beat spectrum
            plot_sim_matrix (Optional[bool]) : Setting this will force plotting the similarity matrix

        EXAMPLE:
             ::
            # To plot the beat spectrum you have a few options:

            # 1) (recommended)
            # set up AudioSignal
            signal = nussl.AudioSignal('path_to_file.wav')

            repet1 = nussl.Repet(signal)

            # plots beat spectrum by default
            repet1.plot('new_beat_spec_plot.png')
        """
        import matplotlib.pyplot as plt
        plt.close('all')
        title = None

        if len(kwargs) != 0:
            if 'title' in kwargs:
                title = kwargs['title']

        beat_spec = self.get_beat_spectrum()
        time_vect = np.linspace(0.0, self.audio_signal.signal_duration, num=len(beat_spec))
        plt.plot(time_vect, beat_spec)
        title = title if title is not None else 'Beat Spectrum for {}'.format(self.audio_signal.file_name)
        plt.title(title)

        plt.xlabel('Time (s)')
        plt.ylabel('Beat Strength')
        plt.grid('on')

        plt.axis('tight')
        plt.savefig(output_file)

    def make_audio_signals(self):
        """ Returns the background and foreground audio signals. You must have run Repet.run() prior
        to calling this function. This function will return None if run() has not been called.

        Returns:
            Audio Signals (List): 2 element list.

                * bkgd: Audio signal with the calculated background track
                * fkgd: Audio signal with the calculated foreground track

        EXAMPLE:
             ::
            # set up AudioSignal object
            signal = nussl.AudioSignal('path_to_file.wav')

            # set up and run repet
            repet = nussl.Repet(signal)
            repet.run()

            # get audio signals (AudioSignal objects)
            background, foreground = repet.make_audio_signals()
        """
        if self.bkgd is None:
            return None

        self.fgnd = self.audio_signal - self.bkgd
        return [self.bkgd, self.fgnd]
