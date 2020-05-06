import numpy as np
import numpy.random as random
import ffmpeg
import librosa
import os
from .audio_signal import AudioSignal
import tempfile
from .augmentation_utils import *

filter_kwargs = {'loglevel': 'quiet'}

def _make_arglist_ffmpeg(lst):
    return "|".join([str(s) for s in lst])

def time_stretch(audio_signal, stretch_factor):
    """
    Linear Stretch on the time axis
    Args: 
        audio_signal: An AudioSignal object
        factor_range: A tuple of length 2. Denotes start and end of possible ranges for factor. 
    Returns:
        stretched_signal: A copy of the original audio_signal, with augmented sources. 
    """
    if not np.issubdtype(type(stretch_factor), np.number) or stretch_factor <= 0:
        raise ValueError("stretch_factor must be a positve scalar")

    sample_rate = audio_signal.sample_rate
    stretched_audio_data = []
    audio_data = audio_signal.audio_data

    for audio_row in audio_signal.get_channels():
        stretched_audio_data.append(librosa.effects.time_stretch(audio_row, stretch_factor))
    stretched_audio_data = np.array(stretched_audio_data)

    # The following line causes a UserWarning.
    # stretched_signal = audio_signal.make_copy_with_audio_data(stretched_audio_data)

    # This one does not
    stretched_signal = AudioSignal(audio_data_array=stretched_audio_data)
    stretched_signal.stft()

    return stretched_signal

def pitch_shift(audio_signal, shift):
    """
    Pitch shift on the frequency axis
    Args: 
        audio_signal: An AudioSignal object
        shift: The number of half-steps to shift the audio. 
            Positive values increases the frequency of the signal
    Returns:
        shifted_signal: A copy of the original audio_signal, with augmented sources. 
    """
    if not isinstance(shift, int):
        raise ValueError("shift must be an integer.")

    sample_rate = audio_signal.sample_rate
    shifted_audio_data = []
    audio_data = audio_signal.audio_data

    for audio_row in audio_signal.get_channels():
        shifted_audio_data.append(librosa.effects.pitch_shift(audio_row, sample_rate, shift))
    shifted_audio_data = np.array(shifted_audio_data)
    shifted_signal = audio_signal.make_copy_with_audio_data(shifted_audio_data)
    shifted_signal.stft()

    return shifted_signal

def low_pass(audio_signal, highest_freq):
    """
    Applies low pass filter. 
    Args: 
        audio_signal: An AudioSignal object
        highest_freq: Threshold for low pass. Should be positive scalar
    Returns:
        augmented_signal: A copy of the original audio_signal, with augmented sources. 
    """
    if not np.issubdtype(type(highest_freq), np.number) or highest_freq <= 0:
        raise ValueError("highest_freq should be positve scalar")
    
    l_stft = audio_signal.stft_data.copy()
    freq_vector = audio_signal.freq_vector
    idx = (np.abs(freq_vector - highest_freq)).argmin()
    l_stft[idx:, :, :] = 0
    augmented_signal = audio_signal.make_copy_with_stft_data(l_stft)
    augmented_signal.istft()

    return augmented_signal

def high_pass(audio_signal, lowest_freq):
    """
    Applies high pass filter
    Args: 
        audio_signal: An AudioSignal object
        highest_freq: Threshold for high pass. Should be positive scalar
    Returns:
        augmented_signal: A copy of the original audio_signal, with augmented sources. 
    """
    if not np.issubdtype(type(lowest_freq), np.number) or lowest_freq <= 0:
        raise ValueError("lowest_freq should be positve scalar")

    h_stft = audio_signal.stft_data.copy()
    freq_vector = audio_signal.freq_vector
    idx = (np.abs(freq_vector - lowest_freq)).argmin()
    h_stft[:idx, :, :] = 0
    augmented_signal = audio_signal.make_copy_with_stft_data(h_stft)
    augmented_signal.istft()

    return augmented_signal

def tremolo(audio_signal, mod_freq, mod_depth):
    """
    Applies tremolo filter
    Args: 
        audio_signal: An AudioSignal object
        mod_freq: Modulation frequency. Must be between 0 and 1.
        mod_depth: Modulation depth. Must be between 0 and 1.
    Returns:
        augmented_item: A copy of the original audio_signal, with augmented sources. 
    """
    if not np.issubdtype(type(mod_freq), np.number) or mod_freq < 0:
        raise ValueError("mod_freq should be positve scalar")
    
    if not np.issubdtype(type(mod_depth), np.number) or mod_depth < 0 or mod_depth > 1:
        raise ValueError("mod_depth should be positve scalar between 0 and 1.")

    audio_tempfile, audio_tempfile_name = \
        save_audio_signal_to_tempfile(audio_signal)
    output_tempfile, output_tempfile_name = \
        make_empty_audio_file()

    output = (ffmpeg
        .input(audio_tempfile_name, **filter_kwargs)
        .filter('tremolo', 
            f = mod_freq,
            d = mod_depth)
        .output(output_tempfile_name)
        .overwrite_output()
        .run()
    )
    augmented_signal = read_audio_tempfile(output_tempfile)
    return augmented_signal


def vibrato(audio_signal, mod_freq, mod_depth):
    """
    Applies vibrato filter
    Args: 
        audio_signal: An AudioSignal object
        mod_freq: Modulation frequency. Must be between 0 and 1.
        mod_depth: Modulation depth. Must be between 0 and 1.
    Returns:
        augmented_item: A copy of the original audio_signal, with augmented sources. 
    """
    if not np.issubdtype(type(mod_freq), np.number) or mod_freq < 0:
        raise ValueError("mod_freq should be positve scalar")
    
    if not np.issubdtype(type(mod_depth), np.number) or mod_depth < 0 or mod_depth > 1:
        raise ValueError("mod_depth should be positve scalar between 0 and 1.")

    audio_tempfile, audio_tempfile_name = \
        save_audio_signal_to_tempfile(audio_signal)
    output_tempfile, output_tempfile_name = \
        make_empty_audio_file()

    output = (ffmpeg
        .input(audio_tempfile_name, **filter_kwargs)
        .filter('vibrato', 
            f = mod_freq,
            d = mod_depth)
        .output(output_tempfile_name)
        .overwrite_output()
        .run()
    )
    augmented_signal = read_audio_tempfile(output_tempfile)
    return augmented_signal

def echo(audio_signal, in_gain, out_gain, delays: list, decays: list):
    """
    Applies echo filter
    Args:
        audio_signal: An AudioSignal object
        in_gain: Input gain of the reflected signal. Must be between 0 and 1.
        out_gain: Output gain of the reflected signal. Must be between 0 and 1.
        delays: A time or list of times in ms between original signal and reflections.
            must be list. Must be scalars in (0 and 90000.0]
        decays: A loudness or list of loudness of reflected signals. 
            must be list. Must be scalars in (0 and 1.0]
    """
    audio_tempfile, audio_tempfile_name = \
        save_audio_signal_to_tempfile(audio_signal)
    output_tempfile, output_tempfile_name = \
        make_empty_audio_file()

    delays_str = _make_arglist_ffmpeg(delays)
    decays_str = _make_arglist_ffmpeg(decays)
    output = (ffmpeg
        .input(audio_tempfile_name, **filter_kwargs)
        .filter('aecho',
        in_gain=in_gain,
        out_gain=out_gain,
        delays=delays_str,
        decays=decays_str)
        .output(output_tempfile_name)
        .overwrite_output()
        .run()
    )

    augmented_signal = read_audio_tempfile(output_tempfile)
    return augmented_signal

def igaussian_filter(audio_signal, factor_range):
    raise NotImplementedError
