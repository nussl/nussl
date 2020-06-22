import numpy as np
import warnings
import ffmpeg
import tempfile
import sox

import .audio_signal as audio_signal
from .constants import LEVEL_MIN, LEVEL_MAX
from .utils import _close_temp_files


"""
Usage Notes about effects.py

The effect functions do not augment an AudioSignal object, but rather 
return a FFmpegFilter or a SoXFilter, which may be called on either a sox.transform.Transformer
or a python-ffmpeg stream, depending on the specific effect. To apply the effect on an AudioSignal, 
build_effects_sox or build_effects_ffmpeg must be called with the AudioSignal, and a list of
SoXFilters or FFmpegFilters, respectively. 

>>> import nussl.effects
>>> tremolo_filter = effects.tremolo(5, .7)
>>> new_signal = effects.build_effects_ffmpeg(audio_signal, [tremolo_filter])

Because of this schema, and the requirement that caller must know which effect is a SoX effect
or a FFmpeg effect, we recommend that all users use the AudioSignal hooks for applying effects
rather than calling any functions in this library. 

This line is equivalent to the above snippet

>>> new_signal = audio_signal.tremolo(5, .7).build_effects()

"""

class FilterFunction:
    """
    The FilterFunction class is an abstract class for functions that take 
    audio processing streams, such as ffmpeg-python and pysndfx
    """

    def __call__(self, stream):
        return self.func(stream)

    def func(self, stream):
        pass


class FFmpegFilter(FilterFunction):
    """
    FFmpegFilter is an object returned by FFmpeg effects in effects.py
    To use them, build_effects_ffmpeg can take a list of effects, and apply them onto an 
    AudioSignal object. 
    """
    def __init__(self, _filter, **filter_kwargs):
        self.filter = _filter
        self.func = lambda stream: stream.filter(_filter, **filter_kwargs)


def build_effects_ffmpeg(audio_signal, filters, silent=False):
    """
    build_effects_ffmpeg takes an AudioSignal object and a list of FFmpegFilter objects
    and sequentially applies each filter to the signal. 
    Args:
        audio_signal (AudioSignal): AudioSignal object
        filters(list): List of FFmpegFilter objects
        silent (bool): If True, suppresses all FFmpeg output. If False, FFmpeg will log information
        with loglevel 'info'
    Returns:
        augmented_signal(AudioSignal): A new AudioSignal object, with the audio data from 
        audio_signal after applying filters

    """

    tmpfiles = []

    input_args = {}
    if silent:
        input_args['loglevel'] = 'quiet'

    with _close_temp_files(tmpfiles):
        curr_tempfile = tempfile.NamedTemporaryFile(suffix=".flac")
        out_tempfile = tempfile.NamedTemporaryFile(suffix=".flac")
        
        tmpfiles.append(curr_tempfile)
        tmpfiles.append(out_tempfile)
        audio_signal.write_audio_to_file(curr_tempfile)

        tmpfiles.append(curr_tempfile)
        stream = ffmpeg.input(curr_tempfile.name, **input_args)
        for _filter in filters:
            stream = _filter(stream)
        (stream
         .output(out_tempfile.name)
         .overwrite_output()
         .run()
         )

        augmented_signal = audio_signal.AudioSignal(path_to_input_file=out_tempfile.name)
    return augmented_signal


class SoXFilter(FilterFunction):
    """
    SoXFilter is an object returned by FFmpeg effects in effects.py
    To use them, build_effects_sox can take a list of effects, and apply them onto an 
    AudioSignal object. 
    """
    def __init__(self, _filter, **filter_kwargs):
        self.filter = _filter
        if _filter == "tempo":
            self.func = lambda tfm: tfm.tempo(**filter_kwargs)
        elif _filter == "pitch":
            self.func = lambda tfm: tfm.pitch(**filter_kwargs)
        else:
            raise ValueError


def build_effects_sox(audio_signal, filters):
    """
    build_effects_sox takes an AudioSignal object and a list of SoXFilter objects
    and sequentially applies each filter to the signal. 
    Args:
        audio_signal (AudioSignal): AudioSignal object
        filters (list): List of SoXFilter objects
    Returns:
        augmented_signal (AudioSignal): A new AudioSignal object, with the audio data from 
        audio_signal after applying filters
    """
    audio_data = audio_signal.audio_data

    tfm = sox.Transformer()
    for _filter in filters:
        tfm = _filter(tfm)
    augmented_data = tfm.build_array(input_array=np.transpose(audio_data), 
        sample_rate_in=audio_signal.sample_rate)

    return audio_signal.make_copy_with_audio_data(np.transpose(augmented_data))


def make_arglist_ffmpeg(lst, sep="|"):
    return sep.join([str(s) for s in lst])


def time_stretch(factor):
    """
    Returns a SoXFilter, when called on an pysndfx stream, will multiply the 
    tempo of the audio by factor. A factor greater than one will shorten the signal, 
    a factor less then one will lengthen the signal, and a factor of 1 will not change the signal.
    Args: 
        factor (float): Scaling factor for tempo change. Must be positive.
    Returns:
        filter (SoXFilter): A SoXFilter object, to be called on an pysndfx stream
    """
    if not np.issubdtype(type(factor), np.number) or factor <= 0:
        raise ValueError("stretch_factor must be a positve scalar")

    return SoXFilter("tempo", factor=factor)


def pitch_shift(shift):
    """
    Returns a SoXFilter, when called on an pysndfx stream, will increase the pitch 
    of the audio by a number of cents, denoted in shift. A positive shift will raise the pitch
    of the signal.
    Args: 
        shift (float): The number of semitones to shift the audio. 
            Positive values increases the frequency of the signal
    Returns:
        filter (SoxFilter): A SoXFilter object, to be called on an pysndfx stream
    """
    if not np.issubdtype(type(shift), np.integer):
        raise ValueError("shift must be an integer.")

    return SoXFilter("pitch", n_semitones=shift)


def _pass_arg_check(freq, poles, width_type, width):
    if not np.issubdtype(type(freq), np.number) or freq <= 0:
        raise ValueError("lowest_freq should be positive scalar")
    if poles not in {1, 2}:
        raise ValueError("poles must be either 1 or 2")
    if width_type not in {"h", "q", "o", "s", "k"}:
        raise ValueError("width_type must be either h, q, o, s, or k.")
    if not np.issubdtype(type(width), np.number) or width <= 0:
        raise ValueError("width should be positive scalar")


def low_pass(freq, poles=2, width_type="h", width=.707):
    """
    https://ffmpeg.org/ffmpeg-all.html#lowpass

    Creates an FFmpegFilter object, which when called on an ffmpeg stream,
    applies a low pass filter to the audio signal.
    Args: 
        freq (float): Threshold for low pass. Should be positive
        poles (int): Number of poles. should be either 1 or 2
        width_type (str): Unit of width for filter. Must be either:
            'h': Hz
            'q': Q-factor
            'o': octave
            's': slope
            'k': kHz
        width (float): Band width in width_type units
    Returns:
        filter (FFmpegFilter): Resulting filter, to be called on a ffmpeg stream
    """
    _pass_arg_check(freq, poles, width_type, width)

    return FFmpegFilter("lowpass", f=freq, p=poles,
                        t=width_type, w=width)


def high_pass(freq, poles=2, width_type="h", width=.707):
    """
    https://ffmpeg.org/ffmpeg-all.html#highpass

    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a high pass filter to the audio signal.
    Args: 
        freq (float): Threshold for high pass. Should be positive scalar
        poles (int): Number of poles. should be either 1 or 2
        width_type (str): Unit of width for filter. Must be either:
            'h': Hz
            'q': Q-factor
            'o': octave
            's': slope
            'k': kHz
        width (float): Band width in width_type units
    Returns:
        filter (FFmpegFilter): Resulting filter, to be called on a ffmpeg stream
    """
    _pass_arg_check(freq, poles, width_type, width)

    return FFmpegFilter("highpass", f=freq, p=poles, t=width_type, w=width)


def tremolo(mod_freq, mod_depth):
    """
    https://ffmpeg.org/ffmpeg-all.html#tremolo

    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a tremolo filter to the audio signal
    Args: 
        mod_freq (float): Modulation frequency. Must be between .1 and 20000.
        mod_depth (float): Modulation depth. Must be between 0 and 1.
    Returns:
        filter (FFmpegFilter): Resulting filter, to be called on a ffmpeg stream
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
        mod_freq (float): Modulation frequency. Must be between .1 and 20000.
        mod_depth (float): Modulation depth. Must be between 0 and 1.
    Returns:
        filter (FFmpegFilter): Resulting filter, to be called on a ffmpeg stream
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
        delays (list of float): delays in ms. Typical Delay is 40ms-6ms
        decays (list of float): decays. Must be between 0 and 1
        speeds (list of float): speeds. Must be between 0 and 1
        depths (list of float): depths. Must be between 0 and 1
        in_gain (float): Proportion of input gain. Must be between 0 and 1
        out_gain (float): Proportion of output gain. Must be between 0 and 1
    Returns:
        filter: Resulting filter, to be called on a ffmpeg stream
    """
    if in_gain > 1 or in_gain < 0 or out_gain > 1 or out_gain < 0:
        raise ValueError("in_gain and out_gain must be between 0 and 1.")

    # Bounds could not be found in ffmpeg docs, but recommendations were given
    if min(delays) < 0 or max(delays) > 1000:
        warnings.warn("One or more delays is far from the "
                      "typical 40-60 ms range. This might produce strange results.", UserWarning)

    if (len(delays) != len(decays) or len(decays)
            != len(speeds) or len(speeds) != len(depths)):
        raise ValueError("Delays, decays, depths, and speeds must all be the same length.")

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
        in_gain (float): Proportion of input gain. Must be between 0 and 1
        out_gain (float): Proportion of output gain. Must be between 0 and 1.
        delay (float): Delay of chorus filter in ms. (Time between original signal and delayed)
        decay (float): Decay of copied signal. Must be between 0 and 1.
        speed (float): Modulation speed of the delayed filter. 
        _type (str): modulation type. Either Triangular or Sinusoidal
            "triangular" or "t" for Triangular
            "sinusoidal" of "s" for sinusoidal
    Returns:
        filter (FFmpegFilter): Resulting filter, to be called on a ffmpeg stream
    """
    if in_gain > 1 or in_gain < 0:
        raise ValueError("in_gain must be between 0 and 1.")
    if out_gain > 1 or out_gain < 0:
        raise ValueError("out_gain must be between 0 and 1.")
    if decay > 1 or decay < 0:
        raise ValueError("delay must be between 0 and 1.")

    allowed_mod_types = {"triangular", "sinusoidal", "t", "s"}
    if _type not in allowed_mod_types:
        raise ValueError(f"_type must be one of the following:\n{allowed_mod_types}")

    # type is reserved word in python, kwarg dict is necessary
    type_kwarg = {
        "type": _type
    }

    return FFmpegFilter("aphaser", in_gain=in_gain,
                        out_gain=out_gain, delay=delay, speed=speed, decay=decay, **type_kwarg)


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
            or speed < .1 or speed > 10
            or phase < 0 or phase > 100):
        raise ValueError("One of the follow values are not in the accepted ranges"
                         f"delay: {delay}\n"
                         f"depth: {depth}\n"
                         f"regen: {regen}\n"
                         f"width: {width}\n"
                         f"speed: {speed}\n"
                         f"phase: {phase}\n"
                         "The following are the bounds for the parameters to flanger()"
                         "0 < delay < 30\n"
                         "0 < depth < 10\n"
                         "-95 < regen < 95\n"
                         "0 < width < 100\n"
                         ".1 < speed < 10\n"
                         "0 < phase < 100"
                         )


def flanger(delay=0, depth=2, regen=0, width=71,
            speed=.5, phase=25, shape="sinusoidal", interp="linear"):
    """
    https://ffmpeg.org/ffmpeg-all.html#flanger

    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a flanger filter to the audio signal.

    Args:
        delay (float): Base delay in ms between original signal and copy.
            Must be between 0 and 30.
        depth (float): Sweep delay in ms. Must be between 0 and 10.
        regen (float): Percentage regeneration, or delayed signal feedback.
            Must be between -95 and 95.
        width (float): Percentage of delayed signal. Must be between 0 and 100.
        speed (float): Sweeps per second. Must be in .1 to 10
        shape (str): Swept wave shape, Must be "triangular" or "sinusoidal".
        phase (float): swept wave percentage-shift for multi channel. Must be between 0 and 100.
        interp (str): Delay Line interpolation. Must be "linear" or "quadratic".
    Returns:
        filter (FFmpegFilter): Resulting filter, to be called on a ffmpeg stream
    """

    _flanger_argcheck(delay, depth, regen, width, speed, phase, shape, interp)

    return FFmpegFilter("flanger", delay=delay,
                        depth=depth, regen=regen, width=width, speed=speed, phase=phase, shape=shape,
                        interp=interp)


def emphasis(level_in, level_out, _type="col", mode='production'):
    """
    https://ffmpeg.org/ffmpeg-all.html#aemphasis

    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a emphasis filter to the audio signal. An emphasis filter boosts frequency ranges the most 
    suspectible to noise in a medium. When restoring sounds from such a medium, a de-emphasis filter is used
    to de-boost boosted frequencies. 
    Args:
        level_in (float): Input gain
        level_out (float): Output gain
        _type (str): physical medium type to convert/deconvert from.
            Must be one of the following: 
            - "col": Columbia 
            - "emi": EMI
            - "bsi": BSI (78RPM)
            - "riaa": RIAA
            - "cd": CD (Compact Disk)
            - "50fm": 50µs FM
            - "75fm": 75µs FM 
            - "50kf": 50µs FM-KF 
            - "75kf": 75µs FM-KF 
        mode (str): Filter mode. Must be one of the following:
            - "reproduction": Apply de-emphasis filter
            - "production": Apply emphasis filter
    Returns:
        filter (FFmpeg Filter): Resulting filter, to be called on a ffmpeg stream
    """
    if level_in < LEVEL_MIN or level_in > LEVEL_MAX \
            or level_out < LEVEL_MIN or level_out > LEVEL_MAX:
        raise ValueError(f"level_in and level_out must both be between {LEVEL_MIN} AND {LEVEL_MAX}")

    allowed_types = {"col", "emi", "bsi", "riaa", "cd",
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
        level_in (float): Input Gain
        mode (str): Mode of compressor operation. Can either be "upward" or "downward". 
        threshold (float): Volume threshold. If a signal's volume is above the threshold,
            gain reduction would apply.
        reduction_ratio (float): Ratio in which the signal is reduced.
        attack (float): Time in ms between when the signal rises above threshold and when 
            reduction is applied
        release (float): Time in ms between when the signal fall below threshold and 
            when reduction is decreased.
        makeup (float): Factor of amplification post-processing
        knee (float): Softens the transition between reduction and lack of thereof. 
            Higher values translate to a softer transition. 
        link (str): Choose average between all channels or mean. String of either
            "average" or "mean.
        detection (str): Whether to process exact signal of the RMS of nearby signals. 
            Either "peak" for exact or "rms".
        mix (float): Proportion of compressed signal in output.
    Returns:
        filter (FFmpegFilter): Resulting filter, to be called on a ffmpeg stream
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
        bands (list of dict): A list of dictionaries, for each band. The required values for each dictionary:
            'chn' (list of int): List of channel numbers to apply filter. Must be list of ints.
            'f' (float): central freqency of band
            'w' (float): Width of the band in Hz
            'g' (float): Band gain in dB
        A dictionary may also contain these values
            't' (int): Set filter type for band. Default 0. Cann be:
                0, for Butterworth
                1, for Chebyshev type 1
                2, for Chebyshev type 2
    Returns:
        filter: Resulting filter, to be called on a ffmpeg stream
    """
    for band in bands:
        for chn in band["chn"]:
            if chn < 0 or not np.issubdtype(type(chn), np.integer):
                raise ValueError("All values in band[\"chn\"] must be positive integers")
        if band["f"] <= 0:
            raise ValueError("band[\"f\"] must be a positive scalar")
        if band["w"] <= 0:
            raise ValueError("band[\"w\"] must be a positive scalar")
        if band["g"] <= 0:
            raise ValueError("band[\"g\"] must be a positive scalar")
        if band.get("t", 0) not in {0, 1, 2}:
            raise ValueError("band[\"t\"] must be in {0, 1, 2}")

    params = make_arglist_ffmpeg([
        make_arglist_ffmpeg([
            f"c{c} f={band['f']} w={band['w']} g={band['g']} t={band.get('t', 0)}"
            for c in band["chn"]
        ])
        for band in bands
    ])

    return FFmpegFilter("anequalizer", params=params)
