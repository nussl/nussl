# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# AudioSignal Basics
# ==================
#
# The nussl.AudioSignal object is the main container for all things related to your audio data. It provides a lot of
# helpful utilities to make it easy to manipulate your audio. Because it is at the heart of all of the source separation
# algorithms in *nussl*, it is crucial to understand how it works. Here we provide a brief introduction to many common
# tasks.
#
# Initialization from a file
# --------------------------
#
# It is easy to initialize an AudioSignal object by loading an audio file from a path. First, let's use the external file zoo to get an audio file to play around with.
#

import nussl
import time
start_time = time.time()

input_file_path = nussl.efz_utils.download_audio_file(
    'schoolboy_fascination_excerpt.wav')

# Now let's initialize it an AudioSignal object with the audio.

signal1 = nussl.AudioSignal(input_file_path)

# Now the AudioSignal object is ready with all of the information about our the signal. Let's also embed the audio signal in a playable object right inside this notebook so we can listen to it! We can also look at its attributes by printing it.

signal1.embed_audio()
print(signal1)

# AudioSignals pack in a lot of useful functionality. For example:

print("Duration: {} seconds".format(signal1.signal_duration))
print("Duration in samples: {} samples".format(signal1.signal_length))
print("Number of channels: {} channels".format(signal1.num_channels))
print("File name: {}".format(signal1.file_name))
print("Full path to input: {}".format(signal1.path_to_input_file))
print("Root mean square energy: {:.4f}".format(signal1.rms().mean()))

# The actual signal data is in ``signal1.audio_data``. It’s just a numpy array, so we can use it as such:

signal1.audio_data

signal1.audio_data.shape

# A few things to note here:
#
# 1. When AudioSignal loads a file, it converts the data to floats between [-1, 1]
# 2. The number of channels is the first dimension, the number of samples is the second.
#
# Initialization from a numpy array
# ---------------------------------
#
# Another common way to initialize an AudioSignal object is by passing in a numpy array. Let’s first make a single channel signal within a numpy array.
#

# +
import numpy as np

sample_rate = 44100  # Hz
dt = 1.0 / sample_rate
dur = 2.0  # seconds
freq = 440  # Hz
x = np.arange(0.0, dur, dt)
x = np.sin(2 * np.pi * freq * x)
# -

# Cool! Now let’s put this into a new AudioSignal object.

signal2 = nussl.AudioSignal(
    audio_data_array=x, sample_rate=sample_rate)
signal2.embed_audio()
print(signal2)

# Note that we had to give a sample rate. If no sample rate is given, then the following is used:

print(f"Default sample rate: {nussl.constants.DEFAULT_SAMPLE_RATE}")

# Other basic manipulations
# -------------------------
#
# If we want to add the audio data in these two signals, it's simple. But there are some gotchas:

signal3 = signal1 + signal2

# Uh oh! I guess it doesn’t make sense to add a stereo signal (``signal1``) and mono signal (``signal2``).
# But if we really want to add these two signals, we can make one of them mono.
#
# *nussl* does this by simply averaging the
# two channels at every sample. We have to explicitly tell *nussl* that we are okay with ``to_mono()``
# changing ``audio_data``. We do that like this:

print(signal1.to_mono(overwrite=True))

# If we hadn’t set ``overwrite=True`` then ``to_mono()`` would just return a new audio signal 
# that is an exact copy of ``signal1`` except it is mono. You will see this pattern 
# come up again. In certain places, :class:`AudioSignal`:'s default behavior is to 
# overwrite its internal data, and in other places the default is to
# **not** overwrite data. See the reference pages for more info. Let's try:

signal3 = signal1 + signal2

# Uh oh! Let's fix this by truncating the longer `signal1` to match `signal2` duration in seconds.

signal1.truncate_seconds(signal2.signal_duration)
print(signal1)

# Now we can finally add them. The adding of these two signals clips, so let's also peak normalize the audio data.

signal3 = signal1 + signal2
signal3.peak_normalize()
signal3.embed_audio()
print(signal3)

# No exceptions this time! Great! ``signal3`` is now a new AudioSignal 
# object. We can similarly subtract two signals.
#
# Let’s write this to a file:

signal3.write_audio_to_file('/tmp/signal3.wav')

# Awesome! Now lets see how we can manipulate the audio in the frequency domain...

end_time = time.time()
time_taken = end_time - start_time
print(f'Time taken: {time_taken:.4f} seconds')
