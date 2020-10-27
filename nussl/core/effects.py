"""
The effect functions do not augment an AudioSignal object, but rather 
return a FFmpegFilter or a SoXFilter, which may be called on either a sox.transform.Transformer
or a python-ffmpeg stream, depending on the specific effect. To apply the effect on an AudioSignal, 
apply_effect_sox or apply_effect_ffmpeg must be called with the AudioSignal, and a list of
SoXFilters or FFmpegFilters, respectively. 

>>> import nussl.effects
>>> tremolo_filter = effects.tremolo(5, .7)
>>> new_signal = effects.apply_effect_ffmpeg(audio_signal, [tremolo_filter])

Because of this schema, and the requirement that caller must know which effect is a SoX effect
or a FFmpeg effect, we recommend that all users use the AudioSignal hooks for applying effects
rather than calling any functions in this library. 

This line is equivalent to the above snippet

>>> new_signal = audio_signal.tremolo(5, .7).apply_effect()

See also: the associated data augmentation tutorial.
"""

import numpy as np
import warnings
import ffmpeg
import tempfile
import copy
try:
    import soxbindings as sox
except Exception:
    import sox

from .constants import LEVEL_MIN, LEVEL_MAX
from .utils import _close_temp_files


class FilterFunction:
    """
    The FilterFunction class is an abstract class for functions that take 
    audio processing streams, such as ffmpeg-python and pysox effects. 

    Don't call this class. It will not do anything. Please use either FFmpegFilter
    or SoXFilter. 

    Attributes:
        filter: Name of filter used
        params: Parameters for filter
    """
    def __init__(self, filter_, **kwargs):
        self.filter = filter_
        self.params = kwargs

    def __str__(self):
        params = ",".join(f"{p}={v}" for p, v in self.params.items())
        return f"{self.filter} (params: {params})"

    def __call__(self, stream):
        return self.func(stream)

    def func(self, stream):
        pass


class FFmpegFilter(FilterFunction):
    """
    FFmpegFilter is an object returned by FFmpeg effects in effects.py
    To use them, apply_effects_ffmpeg can take a list of effects, and apply them onto an 
    AudioSignal object. 
    """
    def __init__(self, filter_, ffmpeg_name=None, **filter_kwargs):
        super().__init__(filter_, **filter_kwargs)
        ffmpeg_name = self.filter if ffmpeg_name is None else ffmpeg_name
        self.func = lambda stream: stream.filter(ffmpeg_name, **filter_kwargs)


def apply_effects_ffmpeg(audio_signal, filters, silent=False):
    """
    apply_effects_ffmpeg takes an AudioSignal object and a list of FFmpegFilter objects
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
    # lazy load to avoid circular import
    from . import AudioSignal

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
        for filter_ in filters:
            stream = filter_(stream)
        (stream
         .output(out_tempfile.name)
         .overwrite_output()
         .run())

        augmented_data = AudioSignal(path_to_input_file=out_tempfile.name).audio_data

    augmented_signal = audio_signal.make_copy_with_audio_data(augmented_data)
    augmented_signal._effects_applied += filters
    return augmented_signal


class SoXFilter(FilterFunction):
    """
    SoXFilter is an object returned by SoX effects in effects.py
    To use them, apply_effects_sox can take a list of effects, and apply them onto an 
    AudioSignal object. 
    """
    def __init__(self, filter_, **filter_kwargs):
        super().__init__(filter_, **filter_kwargs)
        if filter_ == "time_stretch":
            self.func = lambda tfm: tfm.tempo(**filter_kwargs)
        elif filter_ == "pitch_shift":
            self.func = lambda tfm: tfm.pitch(**filter_kwargs)
        else:
            raise ValueError("Unknown SoX effect passed")


def apply_effects_sox(audio_signal, filters):
    """
    apply_effects_sox takes an AudioSignal object and a list of SoXFilter objects
    and sequentially applies each filter to the signal. 
    Args:
        audio_signal (AudioSignal): AudioSignal object
        filters (list): List of SoXFilter objects
    Returns:
        augmented_signal (AudioSignal): A new AudioSignal object, with the audio data from 
        audio_signal after applying filters
    """
    # lazy load to avoid circular import
    from . import AudioSignal

    audio_data = audio_signal.audio_data

    tfm = sox.Transformer()
    for filter_ in filters:
        tfm = filter_(tfm)
    augmented_data = tfm.build_array(
        input_array=audio_data.T,
        sample_rate_in=audio_signal.sample_rate
    ) 

    augmented_signal = audio_signal.make_copy_with_audio_data(augmented_data)
    augmented_signal._effects_applied += filters
    return augmented_signal


def make_arglist_ffmpeg(lst, sep="|"):
    return sep.join([str(s) for s in lst])


def time_stretch(factor, **kwargs):
    """
    Returns a SoXFilter, when called on an pysox stream, will multiply the 
    tempo of the audio by factor. A factor greater than one will shorten the signal, 
    a factor less then one will lengthen the signal, and a factor of 1 will not change the signal.

    It is recommended that users use `AudioSignal.time_stretch` rather than this function.

    This is a SoX effect. Please see 
    https://pysox.readthedocs.io/en/latest/_modules/sox/transform.html#Transformer.tempo
    for details.
    Args: 
        factor (float): Scaling factor for tempo change. Must be positive.
        kwargs: Arugments passed to `sox.transform.tempo`
    Returns:
        filter (SoXFilter): A SoXFilter object, to be called on an pysndfx stream
    """
    if not np.issubdtype(type(factor), np.number) or factor <= 0:
        raise ValueError("stretch_factor must be a positve scalar")

    return SoXFilter("time_stretch", factor=factor, **kwargs)


def pitch_shift(n_semitones, **kwargs):
    """
    Returns a SoXFilter, when called on an pysox stream, will increase the pitch 
    of the audio by a number of semitones, denoted in n_semitones. A positive n_semitones 
    will raise the pitch of the signal.

    It is recommended that users use `AudioSignal.pitch_shift` rather than this function.

    This is a SoX effect. Please see
    https://pysox.readthedocs.io/en/latest/_modules/sox/transform.html#Transformer.pitch
    for details.

    Args: 
        n_semitones (float): The number of semitones to shift the audio.
            Positive values increases the frequency of the signal
        kwargs: Arugments passed to `sox.transform.pitch`
    Returns:
        filter (SoxFilter): A SoXFilter object, to be called on an pysndfx stream
    """
    return SoXFilter("pitch_shift", n_semitones=n_semitones, **kwargs)


def _pass_arg_check(freq, poles, width_type, width):
    if not np.issubdtype(type(freq), np.number) or freq <= 0:
        raise ValueError("lowest_freq should be positive scalar")
    if poles not in {1, 2}:
        raise ValueError("poles must be either 1 or 2")
    if width_type not in {"h", "q", "o", "s", "k"}:
        raise ValueError("width_type must be either h, q, o, s, or k.")
    if not np.issubdtype(type(width), np.number) or width <= 0:
        raise ValueError("width should be positive scalar")


def low_pass(freq, poles=2, width_type="h", width=.707, **kwargs):
    """
    Creates an FFmpegFilter object, which when called on an ffmpeg stream,
    applies a low pass filter to the audio signal.

    It is recommended that users use `AudioSignal.low_pass` rather than this function.

    This is a FFmpeg effect. Please see
    https://ffmpeg.org/ffmpeg-all.html#lowpass
    for details.
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
        kwargs: Arguments passed to `ffmpeg.filter`
    Returns:
        filter (FFmpegFilter): Resulting filter, to be called on a ffmpeg stream
    """
    _pass_arg_check(freq, poles, width_type, width)

    filter_ = FFmpegFilter("low_pass", ffmpeg_name="lowpass", f=freq, p=poles,
                           t=width_type, w=width, **kwargs)
    filter_.params = {"freq": freq, "poles": poles,
                      "width_type": width_type, "width": width, **kwargs}
    
    return filter_


def high_pass(freq, poles=2, width_type="h", width=.707, **kwargs):
    """
    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a high pass filter to the audio signal.

    It is recommended that users use `AudioSignal.high_pass` rather than this function.

    This is a FFmpeg effect. Please see
    https://ffmpeg.org/ffmpeg-all.html#highpass
    for details.
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
        kwargs: Arguments passed to `ffmpeg.filter`
    Returns:
        filter (FFmpegFilter): Resulting filter, to be called on a ffmpeg stream
    """
    _pass_arg_check(freq, poles, width_type, width)

    filter_ = FFmpegFilter("high_pass", ffmpeg_name="highpass", f=freq, p=poles,
                           t=width_type, w=width, **kwargs)
    filter_.params = {"freq": freq, "poles": poles,
                      "width_type": width_type, "width": width, **kwargs}
    return filter_


def tremolo(mod_freq, mod_depth, **kwargs):
    """
    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a tremolo filter to the audio signal

    It is recommended that users use `AudioSignal.tremolo` rather than this function.

    This is a FFmpeg effect. Please see
    https://ffmpeg.org/ffmpeg-all.html#tremolo
    for details.
    Args: 
        mod_freq (float): Modulation frequency. Must be between .1 and 20000.
        mod_depth (float): Modulation depth. Must be between 0 and 1.
        kwargs: Arguments passed to `ffmpeg.filter`
    Returns:
        filter (FFmpegFilter): Resulting filter, to be called on a ffmpeg stream
    """
    if not np.issubdtype(type(mod_freq), np.number) or mod_freq < .1 or mod_freq > 20000:
        raise ValueError("mod_freq should be positive scalar between .1 and 20000")

    if not np.issubdtype(type(mod_depth), np.number) or mod_depth < 0 or mod_depth > 1:
        raise ValueError("mod_depth should be positve scalar between 0 and 1.")

    filter_ = FFmpegFilter("tremolo", f=mod_freq, d=mod_depth, **kwargs)
    filter_.params = {"mod_freq": mod_freq, "mod_depth": mod_depth, **kwargs}
    return filter_


def vibrato(mod_freq, mod_depth, **kwargs):
    """
    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a vibrato filter to the audio signal.

    It is recommended that users use `AudioSignal.vibrato` rather than this function.

    This is a FFmpeg effect. Please see
    https://ffmpeg.org/ffmpeg-all.html#vibrato
    for details.
    Args: 
        mod_freq (float): Modulation frequency. Must be between .1 and 20000.
        mod_depth (float): Modulation depth. Must be between 0 and 1.
        kwargs: Arguments passed to `ffmpeg.filter`
    Returns:
        filter (FFmpegFilter): Resulting filter, to be called on a ffmpeg stream
    """
    if not np.issubdtype(type(mod_freq), np.number) or mod_freq < .1 or mod_freq > 20000:
        raise ValueError("mod_freq should be positve scalar between .1 and 20000")

    if not np.issubdtype(type(mod_depth), np.number) or mod_depth < 0 or mod_depth > 1:
        raise ValueError("mod_depth should be positve scalar between 0 and 1.")

    filter_ = FFmpegFilter("vibrato", f=mod_freq, d=mod_depth, **kwargs)
    filter_.params = {"mod_freq": mod_freq, "mod_depth": mod_depth, **kwargs}
    return filter_


def chorus(delays, decays, speeds, depths,
           in_gain=.4, out_gain=.4, **kwargs):
    """
    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a vibrato filter to the audio signal.

    It is recommended that users use `AudioSignal.chorus` rather than this function.

    This is a FFmpeg effect. Please see
    https://ffmpeg.org/ffmpeg-all.html#chorus
    for details.

    Args:
        delays (list of float): delays in ms. Typical Delay is 40ms-6ms
        decays (list of float): decays. Must be between 0 and 1
        speeds (list of float): speeds. Must be between 0 and 1
        depths (list of float): depths. Must be between 0 and 1
        in_gain (float): Proportion of input gain. Must be between 0 and 1
        out_gain (float): Proportion of output gain. Must be between 0 and 1
        kwargs: Arguments passed to `ffmpeg.filter`
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

    ffmpeg_delays = make_arglist_ffmpeg(delays)
    ffmpeg_speeds = make_arglist_ffmpeg(speeds)
    ffmpeg_decays = make_arglist_ffmpeg(decays)
    ffmpeg_depths = make_arglist_ffmpeg(depths)

    filter_ = FFmpegFilter("chorus", in_gain=in_gain,
                           out_gain=out_gain, delays=ffmpeg_delays,
                           speeds=ffmpeg_speeds, decays=ffmpeg_decays,
                           depths=ffmpeg_depths, **kwargs)

    filter_.params["delays"] = delays
    filter_.params["speeds"] = speeds
    filter_.params["decays"] = decays
    filter_.params["depths"] = depths
    filter_.params.update(kwargs)
    return filter_


def phaser(in_gain=.4, out_gain=.74, delay=3,
           decay=.4, speed=.5, type_="triangular", **kwargs):
    """
    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a phaser filter to the audio signal

    It is recommended that users use `AudioSignal.phaser` rather than this function.

    This is a FFmpeg effect. Please see
    https://ffmpeg.org/ffmpeg-all.html#aphaser
    for details. 
    Args:
        in_gain (float): Proportion of input gain. Must be between 0 and 1
        out_gain (float): Proportion of output gain. Must be between 0 and 1.
        delay (float): Delay of chorus filter in ms. (Time between original signal and delayed)
        decay (float): Decay of copied signal. Must be between 0 and 1.
        speed (float): Modulation speed of the delayed filter. 
        type_ (str): modulation type. Either Triangular or Sinusoidal
            "triangular" or "t" for Triangular
            "sinusoidal" of "s" for sinusoidal
        kwargs: Arguments passed to `ffmpeg.filter`
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
    if type_ not in allowed_mod_types:
        raise ValueError(f"_type must be one of the following:\n{allowed_mod_types}")

    # type is reserved word in python, kwarg dict is necessary
    type_kwarg = {"type": type_}

    filter_ = FFmpegFilter("phaser", ffmpeg_name="aphaser", in_gain=in_gain, out_gain=out_gain,
                           delay=delay, speed=speed, decay=decay, **{**type_kwarg, **kwargs})
    
    filter_.params = copy.deepcopy(filter_.params)
    del filter_.params["type"]
    filter_.params["type_"] = type_
    filter_.params.update(kwargs)
    return filter_
    

def _flanger_argcheck(delay, depth, regen, width,
                      speed, phase, shape, interp):
    error_text = ""
    allowed_shape_types = {"triangular", "sinusoidal"}
    if shape not in allowed_shape_types:
        error_text += f"shape must be one of the following:\n{allowed_shape_types}.\n"
        error_text += f"shape provided is {shape}.\n"
    allowed_interp_types = {"linear", "quadratic"}
    if interp not in allowed_interp_types:
        error_text += f"interp must be one of the following:\n{allowed_interp_types}\n"
        error_text += f"interp provided is {interp}.\n"
    if not 0 <= delay <= 30:
        error_text += f"delay must be in the range 0 <= delay <= 30\n"
        error_text += f"delay provided is {delay}.\n"
    if not -95 <= regen <= 95:
        error_text += f"regen must be in the range -95 <= regen <= 95\n"
        error_text += f"regen provided is {regen}.\n"
    if not 0 <= depth <= 10:
        error_text += f"depth must be in the range 0 <= depth <= 10\n"
        error_text += f"depth provided is {depth}.\n"
    if not .1 <= speed <= 10:
        error_text += f"speed must be in the range .1 <= speed <= 10\n"
        error_text += f"speed provided is {speed}.\n"
    if not 0 <= width <= 100:
        error_text += f"width must be in the range 0 <= width <= 100\n"
        error_text += f"width provided is {width}.\n"
    if not 0 <= phase <= 100:
        error_text += f"phase must be in the range 0 <= phase <= 100\n"
        error_text += f"phase provided is {phase}.\n"
    if error_text:
        raise ValueError(error_text)


def flanger(delay=0, depth=2, regen=0, width=71, speed=.5,
            phase=25, shape="sinusoidal", interp="linear", **kwargs):
    """
    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a flanger filter to the audio signal.

    It is recommended that users use `AudioSignal.flanger` rather than this function.

    This is a FFmpeg effect. Please see
    https://ffmpeg.org/ffmpeg-all.html#flanger
    for details. 
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
        kwargs: Arguments passed to `ffmpeg.filter`
    Returns:
        filter (FFmpegFilter): Resulting filter, to be called on a ffmpeg stream
    """

    _flanger_argcheck(delay, depth, regen, width, speed, phase, shape, interp)

    return FFmpegFilter("flanger", delay=delay, depth=depth, regen=regen, width=width,
                        speed=speed, phase=phase, shape=shape, interp=interp, **kwargs)


def emphasis(level_in, level_out, type_="col", mode='production', **kwargs):
    """
    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a emphasis filter to the audio signal. An emphasis filter boosts frequency ranges 
    the most suspectible to noise in a medium. When restoring sounds from such a medium, a 
    de-emphasis filter is used to de-boost boosted frequencies. 

    It is recommended that users use `AudioSignal.emphasis` rather than this function.

    This is a FFmpeg effect. Please see
    https://ffmpeg.org/ffmpeg-all.html#aemphasis
    Args:
        level_in (float): Input gain
        level_out (float): Output gain
        type_ (str): physical medium type to convert/deconvert from.
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
        kwargs: Arguments passed to `ffmpeg.filter`
    Returns:
        filter (FFmpeg Filter): Resulting filter, to be called on a ffmpeg stream
    """
    if level_in < LEVEL_MIN or level_in > LEVEL_MAX \
            or level_out < LEVEL_MIN or level_out > LEVEL_MAX:
        raise ValueError(f"level_in and level_out must both be between {LEVEL_MIN} AND {LEVEL_MAX}")

    allowed_types = {"col", "emi", "bsi", "riaa", "cd",
                     "50fm", "75fm", "50kf", "75kf"}
    if type_ not in allowed_types:
        raise ValueError(f"Given emphasis filter type is not supported by ffmpeg")
    if mode != "production" and mode != "reproduction":
        raise ValueError(f"mode must be production or reproduction")

    # type is a reserved word in python, so kwarg dict is necessary
    type_kwarg = {'type': type_}

    filter_ = FFmpegFilter("emphasis", ffmpeg_name="aemphasis", level_in=level_in,
                           level_out=level_out, mode=mode, **{**type_kwarg, **kwargs})

    filter_.params = copy.deepcopy(filter_.params)
    del filter_.params["type"]
    filter_.params["type_"] = type_
    filter_.params.update(kwargs)
    return filter_


def _compressor_argcheck(level_in, mode, reduction_ratio,
                         attack, release, makeup, knee, link,
                         detection, mix, threshold):
    error_text = ""
    allowed_mode_types = {"upward", "downward"}
    if mode not in allowed_mode_types:
        error_text += f"shape must be one of the following:\n{allowed_mode_types}.\n"
        error_text += f"shape provided is {mode}.\n"
    allowed_link_types = {"average", "maximum"}
    if link not in allowed_link_types:
        error_text += f"link must be one of the following:\n{allowed_link_types}.\n"
        error_text += f"link provided is {link}.\n"
    allowed_detection_types = {"peak", "rms"}
    if detection not in allowed_detection_types:
        error_text += f"detection must be one of the following:\n{allowed_detection_types}.\n"
        error_text += f"detection provided is {detection}.\n"
    if not LEVEL_MIN <= level_in <= LEVEL_MAX:
        error_text += f"level_in must be in the range {LEVEL_MIN} <= phase <= {LEVEL_MAX}\n"
        error_text += f"level_in provided is {level_in}.\n"
    if not 1 <= reduction_ratio <= 20:
        error_text += f"reduction_ratio must be in the range 1 <= reduction_ratio <= 20\n"
        error_text += f"reduction_ratio provided is {reduction_ratio}.\n"
    if not .01 <= attack <= 2000:
        error_text += f"attack must be in the range 1 <= reduction_ratio <= 2000\n"
        error_text += f"attack provided is {reduction_ratio}.\n"
    if not .01 <= release <= 9000:
        error_text += f"release must be in the range 1 <= release <= 9000\n"
        error_text += f"release provided is {release}.\n"
    if not .1 <= makeup <= 64:
        error_text += f"makeup must be in the range 1 <= makeup <= 64\n"
        error_text += f"makeup provided is {makeup}.\n"
    if not 1 <= knee <= 8:
        error_text += f"knee must be in the range 1 <= knee <= 8\n"
        error_text += f"knee provided is {knee}.\n"
    if not 0 <= mix <= 1:
        error_text += f"mix must be in the range 0 <= mix <= 1\n"
        error_text += f"mix provided is {mix}.\n"
    if not 9.7563e-5 <= threshold <= 1:
        error_text += f"threshold must be in the range 0.000097563 <= threshold <= 1\n"
        error_text += f"threshold provided is {threshold}.\n"
    if error_text:
        raise ValueError(error_text)
    

def compressor(level_in, mode="downward", reduction_ratio=2,
               attack=20, release=250, makeup=1, knee=2.8284, link="average",
               detection="rms", mix=1, threshold=.125, **kwargs):
    """
    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a compressor filter to the audio signal.

    It is recommended that users use `AudioSignal.compressor` rather than this function.

    This is a FFmpeg effect. Please see
    https://ffmpeg.org/ffmpeg-all.html#acompressor
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
        kwargs: Arguments passed to `ffmpeg.filter`
    Returns:
        filter (FFmpegFilter): Resulting filter, to be called on a ffmpeg stream
    """
    _compressor_argcheck(level_in, mode, reduction_ratio,
                         attack, release, makeup, knee, link, detection, mix, threshold)

    filter_ = FFmpegFilter("compressor", ffmpeg_name="acompressor", level_in=level_in,
                           ratio=reduction_ratio, attack=attack, release=release, makeup=makeup,
                           knee=knee, link=link, detection=detection, mix=mix, threshold=threshold,
                           **kwargs)
    filter_.params = copy.deepcopy(filter_.params)
    del filter_.params["ratio"]
    filter_.params["reduction_ratio"] = reduction_ratio
    filter_.params.update(**kwargs)
    return filter_


def equalizer(bands, **kwargs):
    """
    Creates a FFmpegFilter object, when called on an ffmpeg stream,
    applies a equalizer filter to the audio signal.

    It is recommended that users use `AudioSignal.equalizer` rather than this function.

    This is a FFmpeg effect. Please see
    https://ffmpeg.org/ffmpeg-all.html#anequalizer
    Args:
        bands (list of dict): A list of dictionaries, for each band. The required values
            for each dictionary:
            'chn' (list of int): List of channel numbers to apply filter. Must be list of ints.
            'f' (float): central freqency of band
            'w' (float): Width of the band in Hz
            'g' (float): Band gain in dB
        A dictionary may also contain these values
            't' (int): Set filter type for band. Default 0. Cann be:
                0, for Butterworth
                1, for Chebyshev type 1
                2, for Chebyshev type 2
        kwargs: Arguments passed to `ffmpeg.filter`
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

    filter_ = FFmpegFilter("equalizer", ffmpeg_name="anequalizer", params=params, **kwargs)
    filter_.params = {"bands": bands, **kwargs}
    return filter_
