import numpy as np
import numpy.random as random
import ffmpeg
import librosa
import os
from .audio_signal import AudioSignal
import tempfile
from .augmentation_utils import *

# Passing this to an 'ffmpeg.input' will supress all ffmpeg
# output. For troubleshooting, remove this dict from the function args
silent_kwargs = {'loglevel': 'quiet'}

# These values are found in the ffmpeg documentation in filters
# that use the level_in arugment, however, not all filters that
# use this state these bounds in documentationwith pytest.raises(ValueError):
LEVEL_MIN = .015625
LEVEL_MAX = 64

def _make_arglist_ffmpeg(lst):
    return "|".join([str(s) for s in lst])

def time_stretch(audio_signal, stretch_factor):
    """
    Linear Stretch on the time axis
    Args: 
        audio_signal: An AudioSignal object
        factor_range: A tuple of length 2. Denotes start and end of possible ranges for factor. 
    Returns:
        stretched_signal: A copy of the original audio signal, with augmentations applied 
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
        shifted_signal: A copy of the original audio signal, with augmentations applied 
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
        augmented_signal: A copy of the original audio signal, with augmentations applied 
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
        augmented_signal: A copy of the original audio signal, with augmentations applied 
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
    https://ffmpeg.org/ffmpeg-all.html#tremolo

    Applies tremolo filter
    Args: 
        audio_signal: An AudioSignal object
        mod_freq: Modulation frequency. Must be between 0 and 1.
        mod_depth: Modulation depth. Must be between 0 and 1.
    Returns:
        augmented_signal: A copy of the original audio signal, with augmentations applied 
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
        .input(audio_tempfile_name, **silent_kwargs)
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
    https://ffmpeg.org/ffmpeg-all.html#vibrato

    Applies vibrato filter
    Args: 
        audio_signal: An AudioSignal object
        mod_freq: Modulation frequency. Must be between 0 and 1.
        mod_depth: Modulation depth. Must be between 0 and 1.
    Returns:
        augmented_signal: A copy of the original audio signal, with augmentations applied 
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
        .input(audio_tempfile_name, **silent_kwargs)
        .filter('vibrato', 
            f = mod_freq,
            d = mod_depth)
        .output(output_tempfile_name)
        .overwrite_output()
        .run()
    )
    augmented_signal = read_audio_tempfile(output_tempfile)
    return augmented_signal

def chorus(audio_signal, in_gain, out_gain, delays, decays, speeds, depths):
    """
    https://ffmpeg.org/ffmpeg-all.html#chorus

    Applies Chorus Filter
    Args:
        audio_signal: An AudioSignal object
    Returns:
        augmented_signal: A copy of the original audio signal, with augmentations applied
    """
    raise NotImplementedError()

# def echo(audio_signal, in_gain, out_gain, delays: list, decays: list):
#     """
#     https://ffmpeg.org/ffmpeg-all.html#aecho

#     Applies echo filter
#     Args:
#         audio_signal: An AudioSignal object
#         in_gain: Input gain of the reflected signal. Must be between 0 and 1.
#         out_gain: Output gain of the reflected signal. Must be between 0 and 1.
#         delays: A list of times in ms between original signal and reflections.
#             must be list. Must be scalars in (0 and 90000.0]
#         decays: A list of loudness of reflected signals. 
#             must be list. Must be scalars in (0 and 1.0]
#     Returns:
#         augmented_item: A copy of the original audio signal, with augmentations applied 
#     """
#     if len(delays) != len(decays) or len(delays) == 0:
#         raise ValueError("decays and delays must be the same length and cannot be empty")

#     audio_tempfile, audio_tempfile_name = \
#         save_audio_signal_to_tempfile(audio_signal)
#     output_tempfile, output_tempfile_name = \
#         make_empty_audio_file()

#     delays_str = _make_arglist_ffmpeg(delays)
#     decays_str = _make_arglist_ffmpeg(decays)
#     output = (ffmpeg
#         .input(audio_tempfile_name, **filter_kwargs)
#         .filter('aecho',
#         in_gain=in_gain,
#         out_gain=out_gain,
#         delays=delays_str,
#         decays=decays_str)
#         .output(output_tempfile_name)
#         .overwrite_output()
#         .run()
#     )

#     augmented_signal = read_audio_tempfile(output_tempfile)
#     return augmented_signal


def emphasis(audio_signal, level_in, level_out, _type, mode='production'):
    """
    https://ffmpeg.org/ffmpeg-all.html#aemphasis

    Applies the emphasis filter, which either creates or restores 
    signals from various physical mediums. 
    Args:
        audio_signal: An AudioSignal object
        level_in: Input gain
        level_out: Output gain
        _type: medium type to convert/deconvert from 
        mode: reproduction to convert from physical medium, 
            production to convert to physical medium.

    Returns:
        augmented_signal: A copy of the original audio signal, with augmentations applied 
    """
    if level_in < LEVEL_MIN or level_in > LEVEL_MAX \
        or level_out < LEVEL_MIN or level_out > LEVEL_MAX:
        raise ValueError(f"level_in and level_out must both be between {LEVEL_MIN} AND {LEVEL_MAX}")

    audio_tempfile, audio_tempfile_name = \
        save_audio_signal_to_tempfile(audio_signal)
    output_tempfile, output_tempfile_name = \
        make_empty_audio_file()

    emphasis_kwargs = {
        'level_in': level_in,
        'level_out': level_out,
        'mode': mode,
        'type': _type
    }
    output = (ffmpeg
        .input(audio_tempfile_name, **silent_kwargs)
        .filter('aemphasis',
        **emphasis_kwargs)
        .output(output_tempfile_name)
        .overwrite_output()
        .run()
    )

    augmented_signal = read_audio_tempfile(output_tempfile)
    return augmented_signal

def _compressor_argcheck(level_in, mode, reduction_ratio,
    attack, release, makeup, knee, link,
    detection, mix, threshold):

    # The following values are taken from ffmpeg documentation
    if level_in < LEVEL_MIN or level_in > LEVEL_MAX or \
            mode not in {"upward", "downward"} or \
            reduction_ratio < 1 or reduction_ratio > 20 or \
            attack < .01 or attack > 2000 or \
            release < .01 or release > 9000 or \
            makeup < 1 or makeup > 64 or \
            knee < 1 or knee > 8 or \
            link not in {"average", "maximum"} or \
            detection not in {"peak", "rms"} or \
            mix < 0 or mix > 1 or \
            threshold < 9.7563e-5 or threshold > 1:
        raise ValueError("One of the values provided are not within the bounds of the acompressor function"
        f"mode: {mode}"
        f"reduction_ratio: {reduction_ratio}"
        f"attack: {attack}"
        f"release: {release}"
        f"makeup: {makeup}"
        f"knee: {knee}"
        f"link: {link}"
        f"detection: {detection}"
        f"mix: {mix}"
        "See https://ffmpeg.org/ffmpeg-all.html#acompressor for more infomation."
        )

def compressor(audio_signal, level_in, mode="downward", reduction_ratio=2,
    attack=20, release=250, makeup=1, knee=2.8284, link="average",
    detection="rms", mix=1, threshold=.125):
    """
    https://ffmpeg.org/ffmpeg-all.html#acompressor

    Applies the compressor filter to an audio signal
    Args:
        
    Returns:
        augmented_signal: A copy of the original audio signal, with augmentations applied
    """
    _compressor_argcheck(level_in, mode, reduction_ratio,
    attack, release, makeup, knee, link, detection, mix, threshold)

    audio_tempfile, audio_tempfile_name = \
        save_audio_signal_to_tempfile(audio_signal)
    output_tempfile, output_tempfile_name = \
        make_empty_audio_file()

    compressor_kwargs = {
        "level_in": level_in,
    #   "mode": mode, TODO: for some reason the mode arg doesn't work in ffmpeg. figure out why
        "ratio": reduction_ratio,
        "attack": attack,
        "release": release,
        "makeup": makeup,
        "knee": knee,
        "link": link,
        "detection": detection,
        "mix": mix,
        "threshold": threshold
    }

    output = (ffmpeg
        .input(audio_tempfile_name, **silent_kwargs)
        .filter("acompressor", **compressor_kwargs)
        .output(output_tempfile_name)
        .overwrite_output()
        .run()
    )

    augmented_signal = read_audio_tempfile(output_tempfile)
    return augmented_signal
    

def equalizer(audio_signal, bands):
    """
    https://ffmpeg.org/ffmpeg-all.html#anequalizer

    Applies equalizer filter(s) to the audio_signal. 
    Args:
        audio_signal: An AudioSignal object
        bands: A list of dictionaries, for each band. The required values for each dictionary:
            'chn': List of channel numbers to apply filter
            'f': central freqency of band
            'w': Width of the band in Hz
            'g': Band gain in dB
            't': Set filter type for band, optional, can be:
                0, for Butterworth
                1, for Chebyshev type 1
                2, for Chebyshev type 2
    """
    # TODO: Argcheck
    audio_tempfile, audio_tempfile_name = \
        save_audio_signal_to_tempfile(audio_signal)
    output_tempfile, output_tempfile_name = \
        make_empty_audio_file()

    band_string = _make_arglist_ffmpeg([
        _make_arglist_ffmpeg([
            f"c{c} f={band['f']} w={band['w']} g={band['g']} t={band['t']}"
                for c in band["chn"]
        ])
        for band in bands
    ])

    print(band_string)
    output = (ffmpeg
        .input(audio_tempfile_name)
        .filter("anequalizer", params=band_string)
        .output(output_tempfile_name)
        .overwrite_output()
        .run()
    )

    augmented_signal = read_audio_tempfile(output_tempfile)
    return augmented_signal


def igaussian_filter(audio_signal, mean_freq):
    raise NotImplementedError
