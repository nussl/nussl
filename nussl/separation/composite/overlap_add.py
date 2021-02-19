from librosa.core import audio
from .. import SeparationBase
from ... import AudioSignal
import numpy as np
import warnings
import matplotlib.pyplot as plt

class OverlapAdd(SeparationBase):
    def __init__(self, separation_object, window_length=15, find_permutation=False):
        """[summary]

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

    def run(self, *args, **kwargs):
        audio_signal = self.separation_object.audio_signal

        if audio_signal.signal_duration <= self.window_length:
            estimates = self.separation_object()
            self.audio = np.stack([e.audio_data for e in estimates], axis=-1)
            warnings.warn("Signal duration is less than the window length. "
                          "Running algorithm on entire signal and returning.")
            return self.audio
        
        win_samples = int(self.window_length * audio_signal.sample_rate)
        hop_samples = int(self.hop_length * audio_signal.sample_rate)
        audio_signal.zero_pad(hop_samples, hop_samples)
        num_samples = audio_signal.signal_length
        
        audio_data = np.zeros_like(audio_signal.audio_data)
        chunk_starts = np.arange(0, num_samples - hop_samples, hop_samples)
        estimates_from_chunks = []
        
        for i, start_idx in enumerate(chunk_starts):
            end_idx = start_idx + win_samples
            window = AudioSignal.get_window('hanning', win_samples)[None, :]

            # Run it on the chunk
            self.separation_object.audio_signal.set_active_region(start_idx, end_idx)
            estimates_from_chunk = self.separation_object(*args, **kwargs)
            estimates_from_chunks.append({
                'signals': estimates_from_chunk,
                'window': window,
                'start_idx': start_idx,
                'end_idx': end_idx
            })

        audio_data = [audio_data for _ in range(len(estimates_from_chunks[0]['signals']))]
        audio_data = np.stack(audio_data, axis=-1)

        # Combine all the chunks, fixing permutations between them if needed        
        for i, e in enumerate(estimates_from_chunks):
            signals = e['signals']
            if i > 0 and self.find_permutation:
                # do something with permutation finding 
                # (e.g. re-order signals in estimates['signals'], matching to previous set)
                pass
            for i in range(len(signals)):
                windowed_data = e['window'] * signals[i].audio_data
                audio_data[:, e['start_idx']:e['end_idx'], i] += windowed_data
                
        self.audio = audio_data[:, hop_samples:-hop_samples, :]
        self.separation_object.audio_signal.set_active_region_to_default()
        self.separation_object.audio_signal.crop_signal(hop_samples, hop_samples)
        return self.audio

    def make_audio_signals(self):
        estimates = []
        for i in range(self.audio.shape[-1]):
            _estimate = self.audio_signal.make_copy_with_audio_data(
                self.audio[..., i])
            estimates.append(_estimate)
        return estimates

