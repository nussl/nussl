from librosa.core import audio
from .. import SeparationBase
from ... import AudioSignal
import numpy as np
import warnings
import matplotlib.pyplot as plt

class OverlapAdd(SeparationBase):
    def __init__(self, separation_object, window_length=15, hop_length=None, find_permutation=False):
        """Apply overlap/add to a long audio file to separate it in chunks.
        Note that if the hop length is not half the window length, COLA
        may be violated (see https://en.wikipedia.org/wiki/Overlap%E2%80%93add_method 
        for more).

        Args:
            separation_object (SeparationBase): Separation object that overlap and add is applied to.
            window_length (int): Window length of overlap/add window, by default 15 seconds.
            hop_length (int): Hop length of overlap/add window, by default half the window length.
                If the hop length is not half the window length, overlap and add
                may have strange results. 
            find_permutation (bool): Whether or not to find the permutation between chunks before combining.
        """
        super().__init__(separation_object.audio_signal)

        if hop_length is None:
            hop_length = window_length / 2

        self.hop_length = hop_length
        self.window_length = window_length
        self.separation_object = separation_object
        self.find_permutation = find_permutation

    @staticmethod
    def collect_windows(audio_signal, window_length, hop_length):
        """Function which collects overlapping windows from
        an AudioSignal.

        Args:
            audio_signal (AudioSignal): AudioSignal that windows will be collected over.
            window_length (float): Length of window in seconds.
            hop_length (float): How much to shift for each window 
                (overlap is window_length - hop_length) in seconds.

        Returns:
            list: List of audio signal objects.
        """        
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
        """Function which taes a list of windows and overlap adds them into a
        signal the same length as `audio_signal`.

        Args:
            windows (list): List of audio signal objects containing each window, produced by
                `OverlapAdd.collect_windows`.
            audio_signal (AudioSignal): AudioSignal that windows were collected from.
            window_length (float): Length of window in seconds.
            hop_length (float): How much to shift for each window 
                (overlap is window_length - hop_length) in seconds.

        Returns:
            [type]: [description]
        """
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
        """Re-orders estimates according to max signal correlation. 

        Args:
            estimates (list): List of lists containing audio signal estimates.
            find_permutation (bool): Whether or not to permute the lists.

        Returns:
            list: List of lists of audio signals permuted for max signal correlation between
                chunks if find_permutation was true.
        """
        if not find_permutation:
            return estimates
        return estimates

    def process_windows(self, windows, *args, **kwargs):
        """Process the list of windows. By default, the separation object is run
        on each window in sequence, and each window produces estimates. This could
        be overridden in a subclass for more sophisticated processing of
        each window.

        Args:
            windows (list): List of audio signal objects containing each window, produced by
                `OverlapAdd.collect_windows`.

        Returns:
            list: List of lists containing audio signal estimates.
        """
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
