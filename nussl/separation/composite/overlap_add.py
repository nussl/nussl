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
            tuple: Shape of signal after zero-padding.
        """        
        win_samples = int(window_length * audio_signal.sample_rate)
        hop_samples = int(hop_length * audio_signal.sample_rate)
        audio_signal.zero_pad(hop_samples, hop_samples)
        padded_signal_shape = audio_signal.audio_data.shape
        num_samples = audio_signal.signal_length

        win_starts = np.arange(0, num_samples - hop_samples, hop_samples)
        windows = []

        for start_idx in win_starts:
            end_idx = start_idx + win_samples
            audio_signal.set_active_region(start_idx, end_idx)
            window = audio_signal.make_copy_with_audio_data(
                audio_signal.audio_data, verbose=False)
            window.original_signal_length = window.signal_length
            windows.append(window)

        # Make operations on audio signal non-destructive.
        audio_signal.set_active_region_to_default()
        audio_signal.crop_signal(hop_samples, hop_samples)

        return windows, padded_signal_shape

    @staticmethod
    def overlap_and_add(windows, padded_signal_shape, sample_rate, window_length, hop_length):
        """Function which taes a list of windows and overlap adds them into a
        signal the same length as `audio_signal`.

        Args:
            windows (list): List of audio signal objects containing each window, produced by
                `OverlapAdd.collect_windows`.
            padded_signal_shape (tuple): Shape of padded audio signal.
            sample_rate (float): Sample rate of audio signal.
            window_length (float): Length of window in seconds.
            hop_length (float): How much to shift for each window 
                (overlap is window_length - hop_length) in seconds.

        Returns:
            AudioSignal: overlap-and-added signal.
        """
        audio_data = np.zeros(padded_signal_shape)
        win_samples = int(window_length * sample_rate)
        hop_samples = int(hop_length * sample_rate)
        num_samples = padded_signal_shape[-1]

        win_starts = np.arange(0, num_samples - hop_samples, hop_samples)
        window = AudioSignal.get_window('hanning', win_samples)[None, :]

        for i, start_idx in enumerate(win_starts):
            end_idx = start_idx + win_samples
            length = windows[i].signal_length
            audio_data[:, start_idx:end_idx] += window[:, :length] * windows[i].audio_data
        
        signal = AudioSignal(audio_data_array=audio_data, sample_rate=sample_rate)
        signal = signal.crop_signal(hop_samples, hop_samples)
        return signal

    @staticmethod
    def reorder_estimates(estimates, window_length, hop_length, sample_rate, 
                          find_permutation):
        """Re-orders estimates according to max signal correlation. 

        Args:
            estimates (list): List of lists containing audio signal estimates.
            window_length (float): Length of window in seconds.
            hop_length (float): How much to shift for each window 
                (overlap is window_length - hop_length) in seconds.
            sample_rate (int): Sample rate.
            find_permutation (bool): Whether or not to permute the lists.

        Returns:
            list: List of lists of audio signals permuted for max signal correlation between
                chunks if find_permutation was true.
        """
        if not find_permutation:
            return estimates

        def _compute_reordering(x, y):
            overlap_amount = window_length - hop_length
            overlap_amount = int(sample_rate * overlap_amount)

            x = x.mean(0) # to mono
            y = y.mean(0) # to mono
            
            x = x[..., :overlap_amount, :]
            y = y[..., -overlap_amount:, :]

            x -= x.mean(0, keepdims=True)
            y -= y.mean(0, keepdims=True)

            correlations = (x[..., None] * y[..., None, :]).sum(0)
            reorder = np.argmax(correlations, axis=0)
            return reorder

        def _list_to_array(est):
            return np.stack([e.audio_data for e in est], axis=-1)
        
        reordered_estimates = []
        for i, est in enumerate(estimates):
            if i == 0:
                reordered_estimates.append(est)
            else:
                previous_est = _list_to_array(reordered_estimates[i - 1])
                current_est = _list_to_array(est)
                reorder = _compute_reordering(current_est, previous_est)
                reordered_estimates.append([est[r] for r in reorder])

        return reordered_estimates

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

        windows, padded_signal_shape = self.collect_windows(
            audio_signal, self.window_length, self.hop_length)
        estimates = self.process_windows(windows, *args, **kwargs)
        estimates = self.reorder_estimates(
            estimates, self.window_length, self.hop_length, 
            audio_signal.sample_rate, find_permutation=self.find_permutation)

        num_sources = len(estimates[0])
        self.sources = []

        for ns in range(num_sources):
            windows = [est[ns] for est in estimates]
            source = self.overlap_and_add(
                windows, padded_signal_shape, audio_signal.sample_rate, 
                self.window_length, self.hop_length)
            self.sources.append(source)

        return self.sources

    def make_audio_signals(self):
        return self.sources
