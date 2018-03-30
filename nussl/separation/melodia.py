#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings

import numpy as np
from scipy.ndimage.filters import convolve
import vamp

from ..core import constants
import mask_separation_base
import masks
from .. import AudioSignal


class Melodia(mask_separation_base.MaskSeparationBase):
    """Implements melody extraction using Melodia.

    J. Salamon and E. Gómez, "Melody Extraction from Polyphonic Music Signals using Pitch Contour Characteristics",
    IEEE Transactions on Audio, Speech and Language Processing, 20(6):1759-1770, Aug. 2012.

    This needs melodia installed as a vamp plugin, as well as having vampy= for Python installed.

    Parameters:
        input_audio_signal: (AudioSignal object) The AudioSignal object that has the
                            audio data that REPET will be run on.
        high_pass_cutoff: (Optional) (float) value (in Hz) for the high pass cutoff filter.
        do_mono: (Optional) (bool) Flattens AudioSignal to mono before running the algorithm (does not effect the
                        input AudioSignal object)
        use_librosa_stft: (Optional) (bool) Calls librosa's stft function instead of nussl's

    """

    def __init__(self, input_audio_signal, high_pass_cutoff=None, minimum_frequency=55.0,
                 maximum_frequency=1760.0, voicing_tolerance=0.5, minimum_peak_salience=0.0,
                 do_mono=False, use_librosa_stft=constants.USE_LIBROSA_STFT,
                 mask_type=constants.SOFT_MASK, mask_threshold=0.5):

        super(Melodia, self).__init__(input_audio_signal=input_audio_signal, 
                                      mask_type=mask_type, mask_threshold=mask_threshold)
        self.high_pass_cutoff = 100.0 if high_pass_cutoff is None else float(high_pass_cutoff)
        self.background = None
        self.foreground = None
        self.use_librosa_stft = use_librosa_stft
        self.minimum_frequency = float(minimum_frequency)
        self.maximum_frequency = float(maximum_frequency)
        self.voicing_tolerance = float(voicing_tolerance)
        self.minimum_peak_salience = float(minimum_peak_salience)
        self.stft = None
        self.melody = None
        self.melody_signal = None
        self.timestamps = None
        self.foreground_mask = None
        self.background_mask = None

        if do_mono:
            self.audio_signal.to_mono(overwrite=True)

    def extract_melody(self):
        params = {}
        params['minfqr'] = self.minimum_frequency
        params['maxfqr'] = self.maximum_frequency
        params['voicing'] = self.voicing_tolerance
        params['minpeaksalience'] = self.minimum_peak_salience

        try:
            data = vamp.collect(self.audio_signal.audio_data, self.sample_rate,
                                "mtg-melodia:melodia", parameters=params)
        except Exception as e:
            print('**~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~**\n'
                  '*          Are Vamp and Melodia installed correctly?          *\n'
                  '* Check https://bit.ly/2DXbrAk for installation instructions! *\n'
                  '**~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~**')
            raise e

        _, melody = data['vector']
        hop = 128./44100. # hard coded hop in Melodia vamp plugin, converting it to frames.
        timestamps = 8 * hop + np.arange(len(melody)) * hop
        melody[melody < 0] = 0
        self.melody = melody
        self.timestamps = timestamps

    def create_melody_signal(self, num_overtones):
        # Adapted from Melosynth by Justin Salamon: https://github.com/justinsalamon/melosynth

        if self.timestamps[0] > 0:
            estimated_hop = np.median(np.diff(self.timestamps))
            previous_time = max(self.timestamps[0] - estimated_hop, 0)
            self.timestamps = np.insert(self.timestamps, 0, previous_time)
            self.melody = np.insert(self.melody, 0, 0)

        sample_rate = self.audio_signal.sample_rate
        melody_signal = []
        transition_length = .010 # duration for fade in/out and frequency interpretation
        phase = np.zeros(num_overtones)
        previous_frequency = 0
        previous_time = 0

        for time, frequency in zip(self.timestamps, self.melody):
            num_samples = int(np.round((time - previous_time) * sample_rate))
            if num_samples > 0:
                num_transition_samples = float(min(np.round(transition_length * sample_rate),
                                                   num_samples))
                frequency_series = np.ones(num_samples) * previous_frequency

                if previous_frequency > 0 and frequency > 0:
                    frequency_series += np.minimum(np.arange(num_samples) / num_transition_samples, 1) * \
                                        (frequency - previous_frequency)
                elif frequency > 0:
                    frequency_series = np.ones(num_samples) * frequency

                samples = np.zeros(num_samples)
                for overtone in range(num_overtones):
                    overtone_num = overtone + 1
                    phasors = 2 * np.pi * overtone_num * frequency_series / float(sample_rate)
                    phases = phase[overtone] + np.cumsum(phasors)
                    samples += np.sin(phases) / overtone_num
                    phase[overtone] = phases[-1]

                if previous_frequency == 0 and frequency > 0:
                    samples *= np.minimum(np.arange(num_samples) / num_transition_samples, 1)
                elif previous_frequency > 0 and frequency == 0:
                    samples *= np.maximum(1 - np.arange(num_samples) / num_transition_samples, 0)
                elif previous_frequency == 0 and frequency == 0:
                    samples *= 0

                melody_signal.extend(samples)

            previous_frequency = frequency
            previous_time = time

        melody_signal = np.asarray(melody_signal)
        melody_signal *= 0.8 / float(np.max(melody_signal))
        melody_signal = [melody_signal for channel in range(self.audio_signal.num_channels)]
        melody_signal = np.asarray(melody_signal)
        melody_signal = melody_signal[:, 0:self.audio_signal.signal_length]
        melody_signal = AudioSignal(audio_data_array=melody_signal, sample_rate=sample_rate)

        self.melody_signal = melody_signal
        return melody_signal

    def create_harmonic_mask(self, melody_signal):
        normalized_melody_stft = np.abs(melody_signal.stft())
        normalized_melody_stft /= np.max(normalized_melody_stft)

        # Need to threshold the melody stft since the synthesized
        # F0 sequence overtones are at different weights.
        normalized_melody_stft = normalized_melody_stft > 1e-2
        normalized_melody_stft = normalized_melody_stft.astype(float)
        mask = np.empty(self.audio_signal.stft().shape)

        # Smoothing the mask row-wise using a low-pass filter to
        # get rid of discontuinities in the mask.
        kernel =  np.full((1, 20), 1/20.)
        for channel in range(self.audio_signal.num_channels):
            mask[:, :, channel] = convolve(normalized_melody_stft[:, :, channel], kernel)
        return mask

    def run(self):
        """

        Returns:
            foreground (AudioSignal): An AudioSignal object with melodic foreground in
            foreground.audio_data
            (to get the corresponding background run self.make_audio_signals())

        Example:
             ::

        """
        # High pass filter cutoff freq. (in # of freq. bins), +1 to match MATLAB implementation
        self.high_pass_cutoff = int(np.ceil(self.high_pass_cutoff * (self.stft_params.n_fft_bins - 1) /
                                            self.audio_signal.sample_rate)) + 1
        self._compute_spectrum()

        # separate the mixture foreground melody by masking
        if self.melody_signal is None:
            self.extract_melody()
            self.create_melody_signal(100)

        foreground_mask = self.create_harmonic_mask(self.melody_signal)
        foreground_mask[0:self.high_pass_cutoff, :] = 0

        foreground_mask = masks.SoftMask(foreground_mask)
        if self.mask_type == self.BINARY_MASK:
            foreground_mask = foreground_mask.mask_to_binary(self.mask_threshold)

        self.foreground_mask = foreground_mask
        self.background_mask = foreground_mask.invert_mask()

        self.foreground = self.audio_signal.apply_mask(foreground_mask)
        self.foreground.istft(self.stft_params.window_length, self.stft_params.hop_length,
                              self.stft_params.window_type,
                              overwrite=True, use_librosa=self.use_librosa_stft,
                              truncate_to_length=self.audio_signal.signal_length)

        return [self.background_mask, self.foreground_mask]

    def _compute_spectrum(self):
        self.stft = self.audio_signal.stft(overwrite=True, remove_reflection=True,
                                           use_librosa=self.use_librosa_stft)

    def make_audio_signals(self):
        """ Returns the background and foreground audio signals. You must have run FT2D.run() prior
        to calling this function. This function will return None if run() has not been called.

        Returns:
            Audio Signals (List): 2 element list.

                * bkgd: Audio signal with the calculated background track
                * fkgd: Audio signal with the calculated foreground track

        EXAMPLE:
             ::
        """
        if self.foreground is None:
            return None

        background_array = self.audio_signal.audio_data - self.foreground.audio_data
        self.background = self.audio_signal.make_copy_with_audio_data(background_array)
        return [self.background, self.foreground]
