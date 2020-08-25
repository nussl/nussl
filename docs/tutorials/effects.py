# ---
# jupyter:
#   jupytext:
#     formats: py,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Data augmentation
# =================
#
# To create separation models that are more robust, you may want to augment your audio with effects, 
# such as vibrato or compression. These effects are added directly onto AudioSignal objects easily. 
# This tutorial shows example code and results of applying effects to AudioSignal objects.

import nussl
from nussl.datasets.hooks import MUSDB18
import matplotlib.pyplot as plt

musdb = MUSDB18(download=True)
mix_and_sources = musdb.process_item(0)
mix = mix_and_sources["mix"]

# This is what the unaltered track sounds like, as well as the spectrogram.

mix.embed_audio()
nussl.utils.visualize_spectrogram(mix, y_axis="log")

# Effects are stored in an "effects chain", which refers to a queue of effects that will be 
# applied to an AudioSignal object when `apply_effects` is called. We can add an effect to
# the effects chain by using an effects hook, such as `time_stretch`.

# + tags=[]
print(mix.signal_duration)
mix.time_stretch(0.5)
print(mix.signal_duration)
# -

# However, the signal's duration hasn't changed! You will need to call apply_effects to make
# the changes in the signal's effects chains. 

# + tags=[]
new_signal = mix.apply_effects()
print(new_signal.signal_duration)

# This doesn't change the original signal
print(mix.signal_duration)
# -

# Using `apply_effects` will clear out the current effects chain. This behavior can be avoided by setting `reset` to False. Applied effects can be found in `effects_applied`:
#

# + tags=[]
another_signal = mix.apply_effects()
print(another_signal.signal_duration)

print(new_signal.effects_applied)
# -

# To clear out the current effects chain without applying effects, use `reset_effects_chain`.
# It will not reverse effects already applied!
#
# If `apply_effects` is called with empty effects chain, then it returns itself.

another_signal == mix

# You can also chain effects together. This will add a tremolo effect followed by a high pass
# effect to the AudioSignal's effects chains: (Order Matters!)

mix.tremolo(5, .6).high_pass(12000)

# Using overwrite here, we change the audio data of audio_signal, rather than create a new signal. 

mix.apply_effects(overwrite=True)
mix.effects_applied

# If `user_order=False`, FFmpeg effects will be applied AFTER SoX effects, irregardless of the order the hooks 
# are applied. One may want to disable user order for a significant speed up when applying multiple effects. The effects `time_stretch` and `pitch_shift` are SoX effects. All others are FFmpeg effects. For example, the two statements will result in the same altered signal:

signal_1 = mix.pitch_shift(4).tremolo(5, .6).apply_effects(user_order=False)
signal_2 = mix.tremolo(5, .6).pitch_shift(4).apply_effects(user_order=False)
signal_1.effects_applied == signal_2.effects_applied

# Now we will list the effects provided in nussl.

# Get the original signal back
mix_and_sources = musdb.process_item(0)
mix = mix_and_sources["mix"]

# Time stretching
# ---------------
# `AudioSignal.time_stretch(factor)`
#
#
# Stretches the audio signal by a factor of `stretch factor`. For example, when `stretch_factor=2`, then the audio_data becomes two times faster, and when `stretch_factor=.5`, then the audio_data becomes two times slower. 

# Slower
slow_mix = mix.time_stretch(.7).apply_effects()
slow_mix.embed_audio()
nussl.utils.visualize_spectrogram(slow_mix, y_axis="log")

# Faster
fast_mix = mix.time_stretch(1.5).apply_effects()
fast_mix.embed_audio()
nussl.utils.visualize_spectrogram(fast_mix, y_axis="log")

# Pitch shifting
# ---------------
#
# `AudioSignal.pitch_shift(shift)`
#
# Shifts the pitches up by `shift` half steps. If `shift` is negative, the audio is shifted down by `shift` half steps.

# Higher
high_mix = mix.pitch_shift(12).apply_effects()
high_mix.embed_audio()
nussl.utils.visualize_spectrogram(high_mix, y_axis="log")

# Lower
low_mix = mix.pitch_shift(-12).apply_effects()
low_mix.embed_audio()
nussl.utils.visualize_spectrogram(low_mix, y_axis="log")

# Low-pass and high-pass
# ----------------------
#
# `AudioSignal.low_pass(freq, poles=2, width_type="h", width=.707)`
#
# `AudioSignal.high_pass(freq, poles=2, width_type="h", width=.707)`
#
# Implements low and high pass, where `freq` is the thresholds of each filter. `poles` is the number poles in the filter. Each filter has width of `width` units of `width_type`. 
#
# `width_type` can be any of the following:
#    - `h`: Hz
#    - `q`: Q-factor
#    - `o`: octave
#    - `s`: slope
#    - `k`: kHz

# Low pass
low_pmix = mix.low_pass(400, width_type='h', width=20).apply_effects()
low_pmix.embed_audio()
nussl.utils.visualize_spectrogram(low_pmix, y_axis="log")

# High pass
high_pmix = mix.high_pass(2048, width=100).apply_effects()
high_pmix.embed_audio()
nussl.utils.visualize_spectrogram(high_pmix, y_axis="log")

# Tremolo and Vibrato
# ---------------
#
# `AudioSignal.tremolo(mod_freq, mod_depth)`
#
# `AudioSignal.vibrato(mod_freq, mod_depth)`
#
# Applys tremolo/vibrato filter on the audio signal, with a modulation frequency of `mod_freq` Hz, and modulation amplitude of `mod_depth`.

# Tremolo
trem_mix = mix.tremolo(4, .4).apply_effects()
trem_mix.embed_audio()
nussl.utils.visualize_spectrogram(trem_mix, y_axis="log")

# Vibrato
vib_mix = mix.vibrato(4, .4).apply_effects()
vib_mix.embed_audio()
nussl.utils.visualize_spectrogram(vib_mix, y_axis="log")

# Emphasis
# --------
# ```
# AudioSignal.emphasis(level_in, level_out, type_='col', mode='production')
# ```
# An emphasis filter boosts frequency ranges the most suspectible to noise in a medium. When restoring sounds from such a medium, a de-emphasis filter is used to de-boost boosted frequencies. 
# `level_in` and `level_out` are input and output gain respectively. `_type` denotes the medium. If `mode` is `production`, then a emphasis filter is applied. If it is `reproduction`, then a de-emphasis filter is applied. 

level_in = 1
level_out = .8
type_ = "riaa"
riaa_mix = mix.emphasis(level_in, level_out, type_=type_)
riaa_mix.embed_audio()
nussl.utils.visualize_spectrogram(riaa_mix, y_axis="log")

type_ = "cd"
cd_mix = mix.emphasis(level_in, level_out, type_=type_)
cd_mix.embed_audio()
nussl.utils.visualize_spectrogram(cd_mix, y_axis="log")

type_ = "col"
col_mix = mix.emphasis(level_in, level_out, type_=type_)
col_mix.embed_audio()
nussl.utils.visualize_spectrogram(col_mix, y_axis="log")

# Chorus
# ------
#
# ```
# AudioSignal.chorus(delays, decays, speeds, depths, in_gain=.4, out_gain=.4)
# ```
# Applies a chorus filter to the audio signal. `decays`, `delays`, `speeds`, and `depths` are lists, where `decays[i]`, `delays[i]`, `speeds[i]`, and `depths[i]` denote the decay, delay, speed, and depth for chorus filter `i`. Delays are in milliseconds, while decay, speed, and depths must be between 0 and 1, as they are factors of the original signal. `in_gain` and `out_gain` denote input and output gain respectively. 

## Apply two chorus filters
delays = [40, 60]
decays = [.4, .2]
speeds = [.9, .8]
depths = [.8, .6]
in_gain = 1
out_gain = 1
chor_mix = mix.chorus(delays, decays, speeds, depths, in_gain, out_gain).apply_effects()
chor_mix.embed_audio()
nussl.utils.visualize_spectrogram(chor_mix, y_axis="log")

# Phaser
# ------
#
# ```
# AudioSignal.phaser(in_gain=.4, out_gain=.74, delay=3, decay=.4, speed=.5, type_="triangular")
# ```
#
# Applies a phaser effect to the audio signal. `in_gain` and `out_gain` denote input and output gain respectively. `delay` denotes the delay of the copied signal in milliseconds. `decay` and `speed` are factors of the original signal, and must be between 0 and 1. `_type` denotes the type of modulation, which may be either `"triangular"` or `"sinusoidal"`.

in_gain = 1
out_gain = .8
delay = 70
decay = .7
speed = .8
type_ = "triangular"
phas_mix = mix.phaser(in_gain=in_gain, out_gain=out_gain, delay=delay, 
                      decay=decay, speed=speed, type_=type_).apply_effects()
phas_mix.embed_audio()
nussl.utils.visualize_spectrogram(phas_mix, y_axis="log")

# Flanger
# -------
#
# ```
# AudioSignal.flanger(delay=0, depth=2, regen=0, width=71, speed=.5, phase=25, shape="sinusoidal", interp="linear")
# ```
#
# Applies a flanger filter to an AudioSignal.
# `delay` denotes base delay in ms between original signal and copy. Must be between 0 and 30.
# `depth` denotes sweep delay in ms. Must be between 0 and 10.
# `regen` denotes percentage regeneration, or delayed signal feedback. Must be between -95 and 95.
# `width` denotes percentage of delayed signal. Must be between 0 and 100.
# `speed` denotes sweeps per second. Must be in .1 to 10
# `shape` is the swept wave shape, Must be `"triangular"` or `"sinusoidal"`.
# `phase` (is the swept wave percentage-shift for multi channel. Must be between 0 and 100.
# `interp` denotes type of delay Line interpolation. Must be `"linear"` or `"quadratic"`.

delay = 20
depth = 5
regen = 0
width = 60
speed = .7
phase = 30
shape = "sinusoidal"
interp = "linear"
flang_mix = mix.flanger(delay=delay,depth=depth, regen=regen, width=width, 
                speed=speed, phase=phase, shape=shape, interp=interp).apply_effects()
flang_mix.embed_audio()
nussl.utils.visualize_spectrogram(flang_mix, y_axis="log")

# Compressor
# ----------
#
# ```
# AudioSignal.compressor(level_in, mode="downward", reduction_ratio=2,
#                attack=20, release=250, makeup=1, knee=2.8284, link="average",
#                detection="rms", mix=1, threshold=.125)
# ```
# Applies a compressor signal to an AudioSignal. The information about all of these parameters can be found at https://ffmpeg.org/ffmpeg-all.html#acompressor

# +
level_in = 1
mode="downward"
reduction_ratio=2
attack=20
release=250
makeup=1
knee=2.8284
link="average"
detection="rms"
_mix=1
threshold=.125

compress_mix = mix.compressor(level_in=level_in,reduction_ratio=reduction_ratio, attack=attack, 
                    release=release, makeup=makeup,knee=knee, link=link, detection=detection, 
                    mix=_mix, threshold=threshold).apply_effects()
compress_mix.embed_audio()
nussl.utils.visualize_spectrogram(compress_mix, y_axis="log")
# -

# Equalizer
# --------
#
# ```
# AudioSignal.equalizer(bands)
# ```
#
# Applies an equalizer filter to an AudioSignal. `bands` must be a list of dictionaries, one dictionary for each band. 
#
# A dictionary must contain the following key-value pairs. 
#  - `"chn"`: List of channel numbers to apply filter. This is a list of ints. 
#  - `"f"`: Central Frequency
#  - `"w"`: Width of the band in Hz
#  - `"g"`: Band gain in dB
#  
#  
# A dictionary may also contain `"t"`, which denotes the filter type for band. It may be either 0, for Butterworth, 1, for Chebyshev type 1, 2, for Chebyshev type 2. Defaults to 0.

# It may be helpful to know the number of channels first
num_chan = mix.num_channels

bands = [
    {
        "chn": list(range(num_chan)),
        "f": 512,
        "w": 300,
        "g": 5
    },
    {
        "chn": list(range(num_chan)),
        "f": 1024,
        "w": 100,
        "g": 2
        
    }
]
equal_mix = mix.equalizer(bands).apply_effects()
equal_mix.embed_audio()
nussl.utils.visualize_spectrogram(equal_mix, y_axis="log")
