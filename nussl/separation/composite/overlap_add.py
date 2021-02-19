from librosa.core import audio
from .. import SeparationBase
from ... import AudioSignal
import numpy as np
import warnings
import matplotlib.pyplot as plt

class OverlapAdd(SeparationBase):
    def __init__(self, separation_object, window_length=15, find_permutation=False):
        """Apply overlap/add to a long audio file to separate it in chunks.

        Parameters
        ----------
        separation_object : SeparationBase
            Separation object that overlap and add is applied to.
        window_length : int, optional
            Window length of overlap/add window, by default 15 seconds.
            The hop length will be half the window length.
        find_permutation : bool, optional
            Whether or not to find the permutation between chunks before combining.
        """
        super().__init__(separation_object.audio_signal)

        self.hop_length = window_length / 2
        self.window_length = window_length
        self.separation_object = separation_object
        self.find_permutation = find_permutation

    @staticmethod
    def collect_windows(audio_signal, window_length, hop_length):
        win_samples = int(window_length * audio_signal.sample_rate)
        hop_samples = int(hop_length * audio_signal.sample_rate)
        audio_signal.zero_pad(hop_samples, hop_samples)
        num_samples = audio_signal.signal_length

        win_starts = np.arange(0, num_samples - hop_samples, hop_samples)
        windows = []

        for start_idx in win_starts:
            end_idx = start_idx + win_samples
            audio_signal.set_active_region(start_idx, end_idx)
            window = audio_signal.make_copy_with_audio_data(
                audio_signal.audio_data, verbose=False)
            windows.append(window)

        audio_signal.set_active_region_to_default()
        return windows

    @staticmethod
    def overlap_and_add(windows, audio_signal, window_length, hop_length):
        audio_data = np.zeros_like(audio_signal.audio_data)
        win_samples = int(window_length * audio_signal.sample_rate)
        hop_samples = int(hop_length * audio_signal.sample_rate)
        num_samples = audio_signal.signal_length

        win_starts = np.arange(0, num_samples - hop_samples, hop_samples)
        window = AudioSignal.get_window('hanning', win_samples)[None, :]

        for i, start_idx in enumerate(win_starts):
            end_idx = start_idx + win_samples
            length = windows[i].signal_length
            audio_data[:, start_idx:end_idx] += window[:, :length] * windows[i].audio_data
        
        signal = audio_signal.make_copy_with_audio_data(audio_data)
        signal = signal.crop_signal(hop_samples, hop_samples)
        audio_signal.crop_signal(hop_samples, hop_samples)
        return signal

    @staticmethod
    def reorder_estimates(estimates, find_permutation):
        if not find_permutation:
            return estimates
        return estimates

    def process_windows(self, windows, *args, **kwargs):
        estimates = []
        for window in windows:
            self.separation_object.audio_signal = window
            estimates_from_window = self.separation_object(*args, **kwargs)
            estimates.append(estimates_from_window)
        return estimates

    def run(self, *args, **kwargs):
        audio_signal = self.separation_object.audio_signal

        if audio_signal.signal_duration <= self.window_length:
            self.sources = self.separation_object()
            return self.sources

        windows = self.collect_windows(
            audio_signal, self.window_length, self.hop_length)
        estimates = self.process_windows(windows, *args, **kwargs)
        estimates = self.reorder_estimates(
            estimates, find_permutation=self.find_permutation)

        num_sources = len(estimates[0])
        self.sources = []

        for ns in range(num_sources):
            windows = [est[ns] for est in estimates]
            source = self.overlap_and_add(
                windows, audio_signal, self.window_length, self.hop_length)
            self.sources.append(source)

        return self.sources

    def make_audio_signals(self):
        return self.sources
