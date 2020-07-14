"""
Small collection of utilities for altering and remixing
AudioSignal objects. 
"""
import copy

import numpy as np

from . import AudioSignal


def pan_audio_signal(audio_signal, angle_in_degrees):
    """
    Pans an audio signal left or right by the desired number of degrees. This
    returns a copy of the input audio signal.

    Use negative numbers to pan left, positive to pan right. Angles outside of the
    range [-45, 45] raise an error.
    
    Args:
        audio_signal (AudioSignal): Audio signal to be panned.
        angle_in_degrees (float): Angle in degrees to pan by, between -45 and 45.
    
    Raises:
        ValueError: Angles outside of the range [-45, 45] raise an error.
    
    Returns:
        AudioSignal: Audio signal panned by `angle_in_degrees`.
    """
    if angle_in_degrees < -45 or angle_in_degrees > 45:
        raise ValueError(
            "Angle must be between -45 and 45! -45 means "
            "all the way to the left, 45 means all the way "
            "to the right, and 0 is center.")
    panned_signal = copy.deepcopy(audio_signal)
    panned_signal.to_mono()

    panned_signal.audio_data = np.concatenate([
        panned_signal.audio_data, panned_signal.audio_data
    ], axis=0)

    angle = np.deg2rad(angle_in_degrees)
    left_scale = (
        np.sqrt(2)/2 * (np.cos(angle) - np.sin(angle)))
    right_scale = (
        np.sqrt(2)/2 * (np.cos(angle) + np.sin(angle)))
    
    panned_signal.audio_data[0] *= left_scale
    panned_signal.audio_data[1] *= right_scale

    return panned_signal


def delay_audio_signal(audio_signal, delays_in_samples):
    """
    Delays an audio signal by the desired number of samples per channel. This
    returns a copy of the input audio signal.

    Delay must be positive. The end of the audio signal is truncated for that
    channel so that the length remains the same as the original.
    
    Args:
        audio_signal (AudioSignal): Audio signal to be panned.
        delays_in_samples (list of int): List of delays to apply to each channel.
          Should have the same length as number of channels in the AudioSignal.
    
    Raises:
        ValueError: If length of `delays_in_samples`does not match number of channels
          in the audio signal. Or if any items in `delays_in_samples` are float type. Or
          if any delays are negative.
    
    Returns:
        AudioSignal: Audio signal with each channel delayed by the specified number
          samples in `delays_in_samples`.
    """
    if any([not isinstance(x, int) for x in delays_in_samples]):
        raise ValueError("All items in delay_in_samples must be integers.")
    if any([x < 0 for x in delays_in_samples]):
        raise ValueError("All items in delay_in_samples must be non-negative.")
    if len(delays_in_samples) != audio_signal.num_channels:
        raise ValueError(
            "Number of items in delays_in_samples must match number of "
            "channels in audio_signal.")
            
    delayed_signal = copy.deepcopy(audio_signal)

    for i, delay in enumerate(delays_in_samples):
        if delay > 0:
            _audio_data = delayed_signal.audio_data[i]
            original_length = _audio_data.shape[-1]

            _audio_data = np.pad(_audio_data, (delay, 0))
            _audio_data = _audio_data[:original_length]

            delayed_signal.audio_data[i] = _audio_data
    
    return delayed_signal
