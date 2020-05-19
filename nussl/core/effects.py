import numpy as np
import numpy.random as random
from librosa import load
import os
import tempfile
import warnings
import ffmpeg
import tempfile
from pysndfx import AudioEffectsChain

from .constants import LEVEL_MIN, LEVEL_MAX
from .utils import _close_temp_files

class FilterFunction():
    """
    The FilterFunction class is an abstract class for functions that take 
    audio processing streams, such as ffmpeg-python and pysndfx
    """
    def __init__(self, _filter, **filter_kwargs):
        raise NotImplementedError

    def __call__(self, stream):
        return self.func(stream)
        
class FFmpegFilter(FilterFunction):
    def __init__(self, _filter, **filter_kwargs):
        self.filter = _filter
        self.func = lambda stream: stream.filter(_filter, **filter_kwargs)
    
class SoXFilter(FilterFunction):
    def __init__(self, _filter, **filter_kwargs):
        self.filter = _filter
        if _filter == "tempo":
            self.func = lambda stream: stream.tempo(**filter_kwargs)
        elif _filter == "pitch":
            self.func = lambda stream: stream.pitch(**filter_kwargs)
        else:
            raise ValueError
        
def build_effects_ffmpeg(audio_signal, filters, silent=False):
    """
    build_effects_ffmpeg takes an AudioSignal object and a list of FFmpegFilter objects
    and sequentially applies each filter to the signal. 
    Args:
        audio_signal: AudioSignal object
        filters: List of FFmpegFilter objects
        silent: If True, suppresses all FFmpeg output. If False, FFmpeg will log information
        with loglevel 'info'
    Returns:
        augmented_signal: A new AudioSignal object, with the audio data from 
        audio_signal after applying filters
    """

    # lazy load
    from .audio_signal import AudioSignal

    tmpfiles = []

    input_args = {}
    if silent:
        input_args['loglevel'] = 'quiet'

    with _close_temp_files(tmpfiles):
        curr_tempfile = tempfile.NamedTemporaryFile(suffix=".wav")
        audio_signal.write_audio_to_file(curr_tempfile)
        tmpfiles.append(curr_tempfile)
        stream = ffmpeg.input(curr_tempfile.name, **input_args)
        for _filter in filters:
            stream = _filter(stream)
        (stream
            .output(curr_tempfile.name)
            .overwrite_output()
            .run()
        )
        augmented_signal = AudioSignal(path_to_input_file=curr_tempfile.name)
    return augmented_signal

def build_effects_sox(audio_signal, filters):
    """
    build_effects_sox takes an AudioSignal object and a list of SoXFilter objects
    and sequentially applies each filter to the signal. 
    Args:
        audio_signal: AudioSignal object
        filters: List of SoXFilter objects
    Returns:
        augmented_signal: A new AudioSignal object, with the audio data from 
        audio_signal after applying filters
    """
    audio_data = audio_signal.audio_data

    chain = AudioEffectsChain()
    for _filter in filters:
        chain = _filter(chain)
    augmented_data = chain(audio_data)

    return audio_signal.make_copy_with_audio_data(augmented_data)

def make_arglist_ffmpeg(lst, sep="|"):
    return sep.join([str(s) for s in lst])

def time_stretch(factor):
    """
    Returns a SoXFilter, when called on an pysndfx stream, will multiply the 
    tempo of the audio by factor.
    Args: 
        factor: Scaling factor for tempo change. Must be positive.
    Returns:
        filter: A SoXFilter object, to be called on an pysndfx stream
    """
    if not np.issubdtype(type(factor), np.number) or factor <= 0:
         raise ValueError("stretch_factor must be a positve scalar")

    return SoXFilter("tempo", factor=factor)

def pitch_shift(shift):
    """
    Returns a SoXFilter, when called on an pysndfx stream, will increase the pitch 
    of the audio by a number of cents, denoted in shift.
    Args: 
        shift: The number of cents (1/100th of a half step) to shift the audio. 
            Positive values increases the frequency of the signal
    Returns:
        filter: A SoXFilter object, to be called on an pysndfx stream
    """
    if not isinstance(shift, int):
        raise ValueError("shift must be an integer.")

    return SoXFilter("pitch", shift=shift)

def low_pass(freq, poles=2, width_type="h", width=.707):
    """
    https://ffmpeg.org/ffmpeg-all.html#lowpass

    Creates an FFmpegFilter object, which when called on an ffmpeg stream,
    applies a low pass filter to the audio signal.
    Args: 
        freq: Threshold for high pass. Should be positive scalar
        poles: Number of poles. should be either 1 or 2
        width_type: Unit of width for filter. Must be either:
            'h': Hz
            'q': Q-factor
            'o': octave
            's': slope
            'k': kHz
    Returns:
        filter: A FFmpegFilter object, to be called on an ffmpeg stream
    """
    if not np.issubdtype(type(freq), np.number) or freq <= 0:
        raise ValueError("lowest_freq should be positive scalar")
    if poles not in {1, 2}:
        raise ValueError("poles must be either 1 or 2")
    if width_type not in {"h", "q", "o", "s", "k"}:
        raise ValueError("width_type must be either h, q, o, s, or k.")
    if not np.issubdtype(type(width), np.number) or width <= 0:
        raise ValueError("width should be positive scalar")
    # TODO: mix does not work for some reason, despite being listed in ffmpeg docs
    # if mix < 0 or mix > 1:
    #     raise ValueError("mix must be between 0 and 1")


    return FFmpegFilter("lowpass", f=freq, p=poles,
     t=width_type, w=width)

def high_pass(freq, poles=2, width_type="h", width=.707):
    """
    https://ffmpeg.org/ffmpeg-all.html#highpass

    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a high pass filter to the audio signal.
    Args: 
        freq: Threshold for high pass. Should be positive scalar
        poles: Number of poles. should be either 1 or 2
        width_type: Unit of width for filter. Must be either:
            'h': Hz
            'q': Q-factor
            'o': octave
            's': slope
            'k': kHz
    Returns:
        filter: A FFmpegFilter object, to be called on an ffmpeg stream
    """
    if not np.issubdtype(type(freq), np.number) or freq <= 0:
        raise ValueError("lowest_freq should be positive scalar")
    if poles not in {1, 2}:
        raise ValueError("poles must be either 1 or 2")
    if width_type not in {"h", "q", "o", "s", "k"}:
        raise ValueError("width_type must be either h, q, o, s, or k.")
    if not np.issubdtype(type(width), np.number) or width <= 0:
        raise ValueError("width should be positive scalar")

    # TODO: mix does not work for some reason, despite being listed in ffmpeg docs
    # if mix < 0 or mix > 1:
    #     raise ValueError("mix must be between 0 and 1")


    return FFmpegFilter("highpass", f=freq, p=poles, t=width_type, w=width)

def tremolo(mod_freq, mod_depth):
    """
    https://ffmpeg.org/ffmpeg-all.html#tremolo

    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a tremolo filter to the audio signal
    Args: 
        mod_freq: Modulation frequency. Must be between .1 and 20000.
        mod_depth: Modulation depth. Must be between 0 and 1.
    Returns:
        filter: A FFmpegFilter object, to be called on an ffmpeg stream
    """
    if not np.issubdtype(type(mod_freq), np.number) or mod_freq < .1 or mod_freq > 20000:
        raise ValueError("mod_freq should be positive scalar between .1 and 20000")
    
    if not np.issubdtype(type(mod_depth), np.number) or mod_depth < 0 or mod_depth > 1:
        raise ValueError("mod_depth should be positve scalar between 0 and 1.")

    return FFmpegFilter("tremolo", f=mod_freq, d=mod_depth)

def vibrato(mod_freq, mod_depth):
    """
    https://ffmpeg.org/ffmpeg-all.html#vibrato

    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a vibrato filter to the audio signal
    Args: 
        mod_freq: Modulation frequency. Must be between .1 and 20000.
        mod_depth: Modulation depth. Must be between 0 and 1.
    Returns:
        filter: A FFmpegFilter object, to be called on an ffmpeg stream
    """
    if not np.issubdtype(type(mod_freq), np.number) or mod_freq < .1 or mod_freq > 20000:
        raise ValueError("mod_freq should be positve scalar between .1 and 20000")
    
    if not np.issubdtype(type(mod_depth), np.number) or mod_depth < 0 or mod_depth > 1:
        raise ValueError("mod_depth should be positve scalar between 0 and 1.")

    return FFmpegFilter("vibrato", f=mod_freq, d=mod_depth)

def chorus(delays, decays, speeds, depths, 
    in_gain=.4, out_gain=.4):
    """
    https://ffmpeg.org/ffmpeg-all.html#chorus

    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a vibrato filter to the audio signal
    Args:
        input_gain: Proportion of input gain
        output_gain: Proportion of output gain
        delays: list of delays in ms. Typical Delay is 40ms-6ms
        decays: list of decays. Must be between 0 and 1
        speeds: list of speeds. Must be between 0 and 1
        depths: list of depths. Must be between 0 and 1
    Returns:
        filter: A FFmpegFilter object, to be called on an ffmpeg stream
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

    # TODO: Add more argchecks

    delays = make_arglist_ffmpeg(delays)
    speeds = make_arglist_ffmpeg(speeds)
    decays = make_arglist_ffmpeg(decays)
    depths = make_arglist_ffmpeg(depths)
    
    return FFmpegFilter("chorus", in_gain=in_gain, 
        out_gain=out_gain, delays=delays, speeds=speeds, decays=decays, depths=depths)

def phaser(in_gain=.4, out_gain=.74, delay=3, 
        decay=.4, speed=.5, _type="triangular"):
    """
    https://ffmpeg.org/ffmpeg-all.html#aphaser

    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a phaser filter to the audio signal
    Args:
        input_gain: Proportion of input gain. Must be between 0 and 1
        output_gain: Proportion of output gain. Must be between 0 and 1.
        delay: Delay of chorus filter in ms. (Time between original signal and delayed)
        speed: Modulation speed of the delayed filter. 
        _type: modulation type
    Returns:
        filter: A FFmpegFilter object, to be called on an ffmpeg stream
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

    return FFmpegFilter("aphaser", in_gain = in_gain,
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
        "The following are the bounds for the parameters to flanger()"
        "0 < delay < 30\n"
        "0 < depth < 10\n"
        "-95 < regen < 95\n"
        "0 < width < 100\n"
        ".1 < speed < 10\n"
        )

def flanger(delay=0, depth=2, regen=0, width=71, 
    speed=.5, phase=25, shape="sinusoidal", interp="linear"):
    """
    https://ffmpeg.org/ffmpeg-all.html#flanger

    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a flanger filter to the audio signal.

    Args:
        delay: Base delay in ms between original signal and copy.
            Must be between 0 and 30.
        depth: Sweep delay in ms. Must be between 0 and 10.
        regen: Percentage regeneration, or delayed signal feedback.
            Must be between -95 and 95.
        width: Percentage of delayed signal. Must be between 0 and 100.
        speed: Sweeps per second. Must be in .1 to 10
        shape: Swept wave shape, Must be "triangular" or "sinusoidal".
        phase: swept wave percentage-shift for multi channel. Must be between 0 and 100.
        interp: Delay Line interpolation. Must be "linear" or "quadratic".
    Returns:
        filter: A FFmpegFilter object, to be called on an ffmpeg stream
    """

    _flanger_argcheck(delay, depth, regen, width, speed, phase, shape, interp)

    return FFmpegFilter("flanger", delay=delay,
        depth=depth, regen=regen, width=width, speed=speed, phase=phase, shape=shape, 
        interp=interp)

def emphasis(level_in, level_out, _type="col", mode='production'):
    """
    https://ffmpeg.org/ffmpeg-all.html#aemphasis

    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a emphasis filter to the audio signal.
    Args:
        level_in: Input gain
        level_out: Output gain
        _type: medium type to convert/deconvert from 
        mode: reproduction to convert from physical medium, 
            production to convert to physical medium.

    Returns:
        filter: A FFmpegFilter object, to be called on an ffmpeg stream
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

    return FFmpegFilter("aemphasis", level_in=level_in, level_out=level_out, mode=mode, **type_kwarg)

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
        "The following are the bounds for these parameters:\n"
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

def compressor(level_in, mode="downward", reduction_ratio=2,
    attack=20, release=250, makeup=1, knee=2.8284, link="average",
    detection="rms", mix=1, threshold=.125):
    """
    https://ffmpeg.org/ffmpeg-all.html#acompressor

    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a compressor filter to the audio signal.
    Args:
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
        filter: A FFmpegFilter object, to be called on an ffmpeg stream
    """
    _compressor_argcheck(level_in, mode, reduction_ratio,
    attack, release, makeup, knee, link, detection, mix, threshold)

    # TODO: for some reason the mode arg doesn't work in ffmpeg. figure out why 
    return FFmpegFilter("acompressor", level_in=level_in,
        ratio=reduction_ratio, attack=attack, release=release, makeup=makeup,
        knee=knee, link=link, detection=detection, mix=mix, threshold=threshold)
    

def equalizer(bands):
    """
    https://ffmpeg.org/ffmpeg-all.html#anequalizer

    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a equalizer filter to the audio signal.
    Args:
        audio_signal: An AudioSignal object
        bands: A list of dictionaries, for each band. The required values for each dictionary:
            'chn': List of channel numbers to apply filter. Must be list of ints.
            'f': central freqency of band
            'w': Width of the band in Hz
            'g': Band gain in dB
            't': Set filter type for band, optional, can be:
                0, for Butterworth
                1, for Chebyshev type 1
                2, for Chebyshev type 2
    Returns:
        filter: A FFmpegFilter object, to be called on an ffmpeg stream
    """
    for band in bands:
        for chn in band["chn"]:
            if chn < 0 or not isinstance(chn, int):
                raise ValueError("All values in band[\"chn\"] must be positive integers")
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

    return FFmpegFilter("anequalizer", params=params)
