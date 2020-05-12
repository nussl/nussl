import numpy as np
import numpy.random as random
import librosa
import os
from .audio_signal import AudioSignal
import tempfile
from .augmentation_utils import apply_ffmpeg_filter
from .constants import LEVEL_MIN, LEVEL_MAX
import warnings

def make_arglist_ffmpeg(lst):
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

    stretched_audio_data = []

    for audio_row in audio_signal.get_channels():
        stretched_audio_data.append(librosa.effects.time_stretch(audio_row, stretch_factor))
    stretched_audio_data = np.array(stretched_audio_data)

    # The following line causes a UserWarning.
    stretched_signal = audio_signal.make_copy_with_audio_data(stretched_audio_data)

    # Alternative line to suppress UserWarning
    # stretched_signal = AudioSignal(audio_data_array=stretched_audio_data)

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

    for audio_row in audio_signal.get_channels():
        shifted_audio_data.append(librosa.effects.pitch_shift(audio_row, sample_rate, shift))
    shifted_audio_data = np.array(shifted_audio_data)
    shifted_signal = audio_signal.make_copy_with_audio_data(shifted_audio_data)

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
    
    if audio_signal.stft_data is None:
        audio_signal.stft()
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
        raise ValueError("lowest_freq should be positive scalar")

    if audio_signal.stft_data is None:
        audio_signal.stft()
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
        raise ValueError("mod_freq should be positive scalar")
    
    if not np.issubdtype(type(mod_depth), np.number) or mod_depth < 0 or mod_depth > 1:
        raise ValueError("mod_depth should be positve scalar between 0 and 1.")

    return apply_ffmpeg_filter(audio_signal, "tremolo", f=mod_freq, d=mod_depth)

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

    return apply_ffmpeg_filter(audio_signal, "vibrato", f=mod_freq, d=mod_depth)

def chorus(audio_signal, delays, decays, speeds, depths, 
    in_gain=.4, out_gain=.4):
    """
    https://ffmpeg.org/ffmpeg-all.html#chorus

    Applies Chorus Filter(s). Delays, decays, speeds, and depths
    must all lists of the same length.
    Args:
        audio_signal: An AudioSignal object
        input_gain: Proportion of input gain
        output_gain: Proportion of output gain
        delays: list of delays in ms. Typical Delay is 40ms-6ms
        decays: list of decays
        speeds: list of speeds
        depths: list of depths
    Returns:
        augmented_signal: A copy of the original audio signal, with augmentations applied
    """
    if (in_gain > 1 or in_gain < 0 or out_gain > 1 or out_gain < 0):
        raise ValueError("in_gain and out_gain must be between 0 and 1.")

    # Bounds could not be found in ffmpeg docs, but recommendations were given
    if min(delays) < 0 or max(delays) > 1000:
        warnings.warn("One or more delays is far from the " 
            "typical 40-60 ms range. This might produce strange results.", UserWarning)

    if (len(delays) != len(decays) or len(decays)
        !=  len(speeds) or len(speeds) != len(depths)):
        raise ValueError("Delays, decays, depths, and speeds must all be the same length.")
    

    delays = make_arglist_ffmpeg(delays)
    speeds = make_arglist_ffmpeg(speeds)
    decays = make_arglist_ffmpeg(decays)
    depths = make_arglist_ffmpeg(depths)
    
    return apply_ffmpeg_filter(audio_signal, "chorus", in_gain=in_gain, 
        out_gain=out_gain, delays=delays, speeds=speeds, decays=decays, depths=depths)

def phaser(audio_signal, in_gain=.4, out_gain=.74, delay=3, 
        decay=.4, speed=.5, _type="triangular"):
    """
    https://ffmpeg.org/ffmpeg-all.html#aphaser

    Applies Phaser Filter.
    Args:
        audio_signal: An AudioSignal object
        input_gain: Proportion of input gain
        output_gain: Proportion of output gain
        delay: Delay of chorus filter. (Time between original signal and delayed)
        speed: Speed of the delayed filter.
        _type: modulation type
    Returns:
        augmented_signal: A copy of the original audio signal, with augmentations applied
    """
    if (in_gain > 1 or in_gain < 0 or out_gain > 1 or out_gain < 0):
        raise ValueError("in_gain and out_gain must be between 0 and 1.")

    allowed_mod_types = {"triangular", "sinusoidal", "t", "s"}
    if _type not in allowed_mod_types:
        raise ValueError(f"_type must be one of the following:\n{allowed_mod_types}")

    # type is reserved word in python, kwarg dict is necessary
    type_kwarg = {
        "type": _type
    }

    return apply_ffmpeg_filter(audio_signal, "aphaser", in_gain = in_gain,
        out_gain = out_gain, delay = delay, speed = speed, decay = decay, **type_kwarg)


def _flanger_argcheck(delay, depth, regen, width, 
    speed, phase, shape, interp):

    allowed_shape_types = {"triangular", "sinusoidal"}
    if shape not in allowed_shape_types:
        raise ValueError(f"shape must be one of the following:\n{allowed_shape_types}")
    allowed_interp_types = {"linear", "quadratic"}
    if interp not in allowed_interp_types:
        raise ValueError(f"interp must be one of the following:\n{allowed_interp_types}")
    if (delay < 0 or delay > 30 
        or depth < 0 or depth > 10
        or regen < -95 or regen > 95
        or width < 0 or width > 100
        or speed < .1 or speed > 10):
        raise ValueError("One of the follow values are not in the accepted ranges"
        f"delay: {delay}\n"
        f"depth: {depth}\n"
        f"regen: {regen}\n"
        f"width: {width}\n"
        f"speed: {speed}\n"
        )

def flanger(audio_signal, delay=0, depth=2, regen=0, width=71, 
    speed=.5, phase=25, shape="sinusoidal", interp="linear"):
    """
    https://ffmpeg.org/ffmpeg-all.html#flanger

    Args:
        audio_signal: An AudioSignal object
        delay: Base delay in ms between original signal and copy.
            Must be between 0 and 30.
        depth: Sweep delay in ms. Must be between 0 and 10.
        regen: Percentage regeneration, or delayed signal feedback.
            Must be between -95 and 95.
        width: Percentage of delayed signal. Must be between 0 and 100.
        speed: Sweeps per second. Must be in .1 to 10
        shape: Swept wave shape, can be triangular or sinusoidal.
        phase: swept wave percentage-shift for multi channel. Must be between 0 and 100.
        interp: Delay Line interpolation. Must be linear of quadratic
    Returns:
        augmented_signal: A copy of the original audio signal, with augmentations applied
    """

    _flanger_argcheck(delay, depth, regen, width, speed, phase, shape, interp)

    return apply_ffmpeg_filter(audio_signal, "flanger", delay=delay,
        depth=depth, regen=regen, width=width, speed=speed, phase=phase, shape=shape, 
        interp=interp)

def emphasis(audio_signal, level_in, level_out, _type="col", mode='production'):
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
    allowed_types={"col", "emi", "bsi", "riaa", "cd", 
        "50fm", "75fm", "50kf", "75kf"}
    if _type not in allowed_types:
        raise ValueError(f"Given emphasis filter type is not supported by ffmpeg")
    if mode != "production" and mode != "reproduction":
        raise ValueError(f"mode must be production or reproduction")
    
    # type is a reserved word in python, so kwarg dict is necessary
    type_kwarg = {
        'type': _type
    }

    return apply_ffmpeg_filter(audio_signal, "aemphasis", level_in=level_in, level_out=level_out, mode=mode, **type_kwarg)

def _compressor_argcheck(level_in, mode, reduction_ratio,
    attack, release, makeup, knee, link,
    detection, mix, threshold):

    # The following values are taken from ffmpeg documentation
    if (level_in < LEVEL_MIN or level_in > LEVEL_MAX or 
            mode not in {"upward", "downward"} or 
            reduction_ratio < 1 or reduction_ratio > 20 or 
            attack < .01 or attack > 2000 or 
            release < .01 or release > 9000 or 
            makeup < 1 or makeup > 64 or 
            knee < 1 or knee > 8 or 
            link not in {"average", "maximum"} or 
            detection not in {"peak", "rms"} or 
            mix < 0 or mix > 1 or 
            threshold < 9.7563e-5 or threshold > 1):
        raise ValueError("One or more of the values provided are not within the bounds of the acompressor function"
        f"mode: {mode}\n"
        f"reduction_ratio: {reduction_ratio}\n"
        f"attack: {attack}\n"
        f"release: {release}\n"
        f"makeup: {makeup}\n"
        f"knee: {knee}\n"
        f"link: {link}\n"
        f"detection: {detection}\n"
        f"mix: {mix}\n"
        "The following are the bounds for these parameters:"
        f"{LEVEL_MIN} < level_in < {LEVEL_MAX}\n"
        "mode must be in {'upward, 'downward'}\n"
        "1 < reduction_ratio < 20\n"
        ".01 < attack < 2000\n"
        ".01 < release < 9000\n"
        "1 < makeup < 64\n"
        "1 < knee < 8\n"
        "link must be in {'average', 'maximum'}\n"
        "detection must be in {'peak', 'rms'}\n"
        "0 < mix < 1\n"
        " .000097563 < threshold < 1\n"
        )

def compressor(audio_signal, level_in, mode="downward", reduction_ratio=2,
    attack=20, release=250, makeup=1, knee=2.8284, link="average",
    detection="rms", mix=1, threshold=.125):
    """
    https://ffmpeg.org/ffmpeg-all.html#acompressor

    Applies the compressor filter to an audio signal.
    See ffmpeg documentation for bounds
    Args:
        audio_signal: An AudioSignal object
        level_in: Input Gain
        mode: Mode of compressor operation. Can either be "upward" or "downward". 
        threshold: Volume threshold. If a signal's volume is above the threshold,
            gain reduction would apply.
        reduction_ratio: Ratio in which the signal is reduced.
        attack: Time in ms between when the signal rises above threshold and when 
            reduction is applied
        release: Time in ms between when the signal fall below threshold and 
            when reduction is decreased.
        makeup: Factor of amplification post-processing
        knee: Softens the transition between reduction and lack of thereof. 
            Higher values translate to a softer transition. 
        link: Choose average between all channels or mean. String of either
            "average" or "mean.
        detection: Whether to process exact signal of the RMS of nearby signals. 
            Either "peak" for exact or "rms".
        mix: Proportion of compressed signal in output.
    Returns:
        augmented_signal: A copy of the original audio signal, with augmentations applied
    """
    _compressor_argcheck(level_in, mode, reduction_ratio,
    attack, release, makeup, knee, link, detection, mix, threshold)

    # TODO: for some reason the mode arg doesn't work in ffmpeg. figure out why 
    return apply_ffmpeg_filter(audio_signal, "acompressor", level_in=level_in,
        ratio=reduction_ratio, attack=attack, release=release, makeup=makeup,
        knee=knee, link=link, detection=detection, mix=mix, threshold=threshold)
    

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
    for band in bands:
        if max(band["chn"]) > audio_signal.num_channels:
            raise ValueError("band[\"chn\"] contains a number greater than number of channels")
        if band["f"] <= 0:
            raise ValueError("band[\"f\"] must be a positive scalar")
        if band["w"] <= 0:
            raise ValueError("band[\"w\"] must be a positive scalar")
        if band["g"] <= 0:
            raise ValueError("band[\"g\"] must be a positive scalar")
        if band["t"] not in {0, 1, 2}:
            raise ValueError("band[\"t\"] must be in {0, 1, 2}")
        
    params = make_arglist_ffmpeg([
            make_arglist_ffmpeg([
                f"c{c} f={band['f']} w={band['w']} g={band['g']} t={band['t']}"
                    for c in band["chn"]
            ])
            for band in bands
        ])

    return apply_ffmpeg_filter(audio_signal, "anequalizer", params)


