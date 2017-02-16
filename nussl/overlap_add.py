#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import triang

import spectral_utils
import separation_base
import constants
from audio_signal import AudioSignal
from repet import Repet
from repet_sim import RepetSim
from ft2d import FT2D

class OverlapAdd(separation_base.SeparationBase):
    """Implements foreground/background separation using overlap/add with an arbitrary foreground/background separation scheme in nussl. Currently supports 'REPET', 'REPET-SIM', and 'FT2D'.

    Parameters:
        input_audio_signal: (AudioSignal object) The AudioSignal object that has the
                            audio data that REPET will be run on.
        high_pass_cutoff: (Optional) (float) value (in Hz) for the high pass cutoff filter.
        do_mono: (Optional) (bool) Flattens AudioSignal to mono before running the algorithm (does not effect the
                        input AudioSignal object)
        use_librosa_stft: (Optional) (bool) Calls librosa's stft function instead of nussl's

    """
    def __init__(self, input_audio_signal, separation_method = None, window_size = None, hop_size = None, window_type = None, do_mono=False, use_librosa_stft=constants.USE_LIBROSA_STFT):
        super(OverlapAdd, self).__init__(input_audio_signal=input_audio_signal)
        self.background = None
        self.foreground = None
        self.use_librosa_stft = use_librosa_stft
        self.window_size = 24 if window_size is None else window_size
        self.hop_size = 12 if hop_size is None else hop_size
        self.window_type = "triangular" if window_type is None else window_type
        self.separation_method = separation_method
        
        if self.separation_method not in ['REPET', 'REPET-SIM', 'FT2D']:
            raise ValueError("Cannot run this without a separation method! Choose one from [REPET, REPET-SIM, FT2D")

        if do_mono:
            self.audio_signal.to_mono(overwrite=True)

    def separate(self):
        if self.separation_method == 'REPET':
            repet = Repet(self.audio_signal)
            background = repet.run()
            return background
        elif self.separation_method == 'REPET-SIM':
            repet_sim = RepetSim(self.audio_signal)
            background = repet_sim.run()
            return background
        elif self.separation_method == 'FT2D':
            ft2d = FT2D(self.audio_signal)
            background = ft2d.run()
            return background

    def run(self):
        """

        Returns:
            background (AudioSignal): An AudioSignal object with repeating background in background.audio_data
            (to get the corresponding non-repeating foreground run self.make_audio_signals())

        Example:
             ::

        """
        
        num_samples = self.audio_signal.signal_length
        window_samples = int(np.round(self.audio_signal.sample_rate * self.window_size))
        hop_samples = int(np.round(self.audio_signal.sample_rate * self.hop_size))
        overlap_samples = window_samples - hop_samples

        if num_samples < window_samples + hop_samples:
            return self.separate()
        else:
            num_segments = int(1 + np.floor((num_samples - window_samples) / float(hop_samples)))
            overlap_window = triang(2*overlap_samples)
            overlap_window = np.vstack([overlap_window for i in range(self.audio_signal.num_channels)])

        background = np.zeros((self.audio_signal.num_channels, num_samples))

        for num_segment in range(num_segments):
            if num_segment < num_segments - 1:
                start = num_segment * hop_samples
                end = start + window_samples
            else:
                start = num_segment * hop_samples
                end = num_samples
            self.audio_signal.set_active_region(start, end)
            bg = self.separate()
            if num_segment == 0:
                background[:, 0:bg.signal_length] += bg.audio_data
            else:
                if num_segment == num_segments - 1:
                    overlap_window[:, overlap_samples:] = 1
                start = num_segment * hop_samples
                length = min([bg.signal_length, window_samples])
                background[:, start:start+overlap_samples] = np.multiply(background[:, start:start+overlap_samples], overlap_window[:, overlap_samples:])
                bg.audio_data = np.multiply(bg.audio_data[:, 0:length], overlap_window[:, 0:length])
                background[:, start:start + length] += bg.audio_data[:, 0:length]
        self.audio_signal.set_active_region_to_default()
        self.background = AudioSignal(audio_data_array = background, sample_rate=self.audio_signal.sample_rate)
        return self.background
    
    def make_audio_signals(self):
        """ Returns the background and foreground audio signals. You must have run OverlapAdd.run() prior
        to calling this function. This function will return None if run() has not been called.

        Returns:
            Audio Signals (List): 2 element list.

                * bkgd: Audio signal with the calculated background track
                * fkgd: Audio signal with the calculated foreground track

        EXAMPLE:
             ::
        """
        if self.background is None:
            return None

        self.foreground = self.audio_signal - self.background
        self.foreground.sample_rate = self.audio_signal.sample_rate
        return [self.background, self.foreground]
