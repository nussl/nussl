from .. import SeparationBase
from ... import AudioSignal
import numpy as np
import tqdm

class OverlapAdd(SeparationBase):
    def __init__(self, separation_object, window_duration=15, hop_duration=None, find_permutation=False, verbose=False):
        """Apply overlap/add to a long audio file to separate it in chunks.
        Note that if the hop length is not half the window length, COLA
        may be violated (see https://en.wikipedia.org/wiki/Overlap%E2%80%93add_method 
        for more).

        Args:
            separation_object (SeparationBase): Separation object that overlap and add is applied to.
            window_duration (int): Window duration of overlap/add window, by default 15 seconds.
            hop_duration (int): Hop duration of overlap/add window, by default half the window duration.
                If the hop duration is not half the window duration, overlap and add
                may have strange results. 
            find_permutation (bool): Whether or not to find the permutation between chunks before combining.
            verbose (bool): Whether or not to show a progress bar as file is being separated.
        """
        if hop_duration is None:
            hop_duration = window_duration / 2

        self.hop_duration = hop_duration
        self.window_duration = window_duration
        self.separation_object = separation_object
        self.find_permutation = find_permutation
        self.verbose = verbose

        super().__init__(separation_object.audio_signal)

    def _preprocess_audio_signal(self):
        """If the audio signal of OverlapAdd is set, the separation object's audio signal
        should also be set.
        """
        self.separation_object.audio_signal = self.audio_signal

    @staticmethod
    def collect_windows(audio_signal, window_duration, hop_duration):
        """Function which collects overlapping windows from
        an AudioSignal.

        Args:
            audio_signal (AudioSignal): AudioSignal that windows will be collected over.
            window_duration (float): Length of window in seconds.
            hop_duration (float): How much to shift for each window 
                (overlap is window_duration - hop_duration) in seconds.

        Returns:
            list: List of audio signal objects.
            tuple: Shape of signal after zero-padding.
        """        
        win_samples = int(window_duration * audio_signal.sample_rate)
        hop_samples = int(hop_duration * audio_signal.sample_rate)
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
    def overlap_and_add(windows, padded_signal_shape, sample_rate, window_duration, hop_duration):
        """Function which takes a list of windows and overlap adds them into a
        signal the same length as `audio_signal`.

        Args:
            windows (list): List of audio signal objects containing each window, produced by
                `OverlapAdd.collect_windows`.
            padded_signal_shape (tuple): Shape of padded audio signal.
            sample_rate (float): Sample rate of audio signal.
            window_duration (float): Length of window in seconds.
            hop_duration (float): How much to shift for each window 
                (overlap is window_duration - hop_duration) in seconds.

        Returns:
            AudioSignal: overlap-and-added signal.
        """
        audio_data = np.zeros(padded_signal_shape)
        win_samples = int(window_duration * sample_rate)
        hop_samples = int(hop_duration * sample_rate)
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
    def reorder_estimates(estimates, window_duration, hop_duration, sample_rate, 
                          find_permutation):
        """Re-orders estimates according to max signal correlation. 

        Args:
            estimates (list): List of lists containing audio signal estimates as AudioSignal objects.
            window_duration (float): Length of window in seconds.
            hop_duration (float): How much to shift for each window 
                (overlap is window_duration - hop_duration) in seconds.
            sample_rate (int): Sample rate.
            find_permutation (bool): Whether or not to permute the lists.

        Returns:
            list: List of lists of audio signals permuted for max signal correlation between
                chunks if find_permutation was true.
        """
        if not find_permutation:
            return estimates

        def _compute_reordering(x, y):
            overlap_amount = window_duration - hop_duration
            overlap_amount = int(sample_rate * overlap_amount)

            # Shape is (ch, samples, sources)
            x = x.mean(0) # to mono
            y = y.mean(0) # to mono
            # Shape is (samples, sources)
            
            # Take first half of x, last half of y.
            x = x[..., :overlap_amount, :]
            y = y[..., -overlap_amount:, :]
            
            # Zero-mean both sources.
            x -= x.mean(0, keepdims=True)
            y -= y.mean(0, keepdims=True)

            # Compute cross-correlation between each source.
            # (samples, sources, 1) * (samples, 1, sources) -> (samples, sources, sources)
            # .sum(0) -> (sources, sources)
            correlations = (x[..., None] * y[..., None, :]).sum(0)

            # argmax (sources, sources) matrix to find best permutation.
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

        pbar = range(len(windows))
        if self.verbose:
            pbar = tqdm.trange(len(windows))

        for i in pbar:
            window = windows[i]
            self.separation_object.audio_signal = window
            estimates_from_window = self.separation_object(*args, **kwargs)
            estimates.append(estimates_from_window)
        return estimates

    def run(self, *args, **kwargs):
        audio_signal = self.separation_object.audio_signal

        if audio_signal.signal_duration <= self.window_duration:
            self.sources = self.separation_object()
            return self.sources

        windows, padded_signal_shape = self.collect_windows(
            audio_signal, self.window_duration, self.hop_duration)
        estimates = self.process_windows(windows, *args, **kwargs)
        estimates = self.reorder_estimates(
            estimates, self.window_duration, self.hop_duration, 
            audio_signal.sample_rate, find_permutation=self.find_permutation)

        num_sources = len(estimates[0])
        self.sources = []

        for ns in range(num_sources):
            windows = [est[ns] for est in estimates]
            source = self.overlap_and_add(
                windows, padded_signal_shape, audio_signal.sample_rate, 
                self.window_duration, self.hop_duration)
            self.sources.append(source)

        return self.sources

    def make_audio_signals(self):
        return self.sources
