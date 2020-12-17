"""
Small collection of utilities for altering and remixing
AudioSignal objects. 
"""
import copy
import numpy as np
from . import AudioSignal
from .constants import MIN_LOUDNESS, MAX_LOUDNESS
import scipy
import warnings

def convolve(signal, other, method='auto', normalize=True, 
             scale=True):
    """
    Convolves signal one with signal two. There are three
    cases:
    
    1. s1 is multichannel and s2 is mono.
    -> s1's channels will all be convolved with s2.
    2. s1 is mono and s2 is multichannel.
    -> s1 will be convolved with each channel of s2.
    3. s1 and s2 are both multichannel.
    -> each channel will be convolved with the matching 
        channel. If they don't have the same number of
        channels, an error will be thrown.

    Args:
        other: AudioSignal with which to convolve the two signals.
        method: A string indicating which method to use to calculate the convolution. Options are:
            direct: The convolution is determined directly from sums, the definition of convolution.
            fft: The Fourier Transform is used to perform the convolution by calling fftconvolve.
            auto: Automatically chooses direct or Fourier method based on an estimate of which is faster (default).
        normalize: Whether to apply a normalization factor which will prevent clipping. Defaults to True.
        scale: Whether to scale the output convolved signal to have the same max as the input, so they are of roughly equal loudness.
    
    Example:

    >>> import nussl
    >>> path = nussl.efz_utils.download_audio_file('schoolboy_fascination_excerpt.wav')
    >>> sig = nussl.AudioSignal(path)
    >>> path = nussl.efz_utils.download_audio_file('h072_Bar_2txts.wav')
    >>> ir = nussl.AudioSignal(path)
    >>> convolved = nussl.mixing.convolve(sig, ir)
    >>> # Or apply with fluent interface
    >>> convolved = sig.convolve(ir) 
    
    """
    output = []
    factor = 1.0

    signal = copy.deepcopy(signal)
    other = copy.deepcopy(other)

    if normalize: factor = np.sum(np.abs(other.audio_data) ** 2)
    
    if signal.num_channels != 1 and other.num_channels != 1:
        if signal.num_channels != other.num_channels:
            raise RuntimeError(
                "If both signals are multichannel, they must have the " 
                "same number of channels!")
        for s1_ch, s2_ch in zip(signal.get_channels(), other.get_channels()):
            convolved_ch = scipy.signal.convolve(
                s1_ch, s2_ch / factor, mode='full', method=method)
            output.append(convolved_ch)
    else:
        for i, s1_ch in enumerate(signal.get_channels()):
            for j, s2_ch in enumerate(other.get_channels()):
                convolved_ch = scipy.signal.convolve(
                    s1_ch, s2_ch / factor, mode='full', method=method)
                output.append(convolved_ch)
    
    output = np.array(output)
    if scale:
        max_output = np.abs(output).max()
        max_input = np.abs(signal.audio_data).max()
        scale_factor = max_input / max_output
        output *= scale_factor
    
    convolved_signal = signal.make_copy_with_audio_data(
        output, verbose=False)
    convolved_signal.truncate_samples(signal.signal_length)
            
    return convolved_signal

def mix_audio_signals(fg_signal, bg_signal, snr=10):
    """
    Mixes noise with signal at specified 
    signal-to-noise ratio. Returns the mix, and the altered foreground
    and background sources.

    Args:
        fg_signal: AudioSignal object, will be the louder signal (or quieter if snr is negative).
        bg_signal: AudioSignal object, will be the quieter signal (or louder if snr is negative).
        snr: Signal-to-noise ratio in decibels to mix at.
        inplace: Whether or not to do this operation in place.
        return_sources: Whether to return the sources or return the mix.
    """
    fg_signal = copy.deepcopy(fg_signal)
    bg_signal = copy.deepcopy(bg_signal)

    pad_len = max(0, fg_signal.signal_length - bg_signal.signal_length)
    bg_signal.zero_pad(0, pad_len)
    bg_signal.truncate_samples(fg_signal.signal_length)

    n_loudness = max(MIN_LOUDNESS, bg_signal.loudness())
    loudness = max(MIN_LOUDNESS, fg_signal.loudness())
    
    if loudness - snr < MIN_LOUDNESS:
        old_snr = snr
        snr = loudness - MIN_LOUDNESS
        warnings.warn(
            f"SNR puts loudness below minimum ({MIN_LOUDNESS} dB), "
            f"clipping SNR from {old_snr} to approx. {snr}. ",
            UserWarning
        )

    if loudness - snr > MAX_LOUDNESS:
        old_snr = snr
        snr = loudness - MAX_LOUDNESS
        warnings.warn(
            f"SNR puts loudness above maximum ({MAX_LOUDNESS} dB), "
            f"clipping SNR from {old_snr} to approx. {snr}. ",
            UserWarning
        )

    t_loudness = loudness - snr
    gain = np.mean(t_loudness - n_loudness)
    gain = np.exp(gain * np.log(10) / 20)
    bg_signal = bg_signal * gain
    premix = fg_signal + bg_signal
    
    peak_gain = np.abs(premix.audio_data).max()
    if peak_gain > 1:
        fg_signal = fg_signal / peak_gain
        bg_signal = bg_signal / peak_gain
        premix = premix / peak_gain
    
    return premix, fg_signal, bg_signal

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
