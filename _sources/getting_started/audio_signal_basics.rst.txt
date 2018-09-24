.. _audio_signal_basics:

==================
AudioSignal Basics
==================

The :class:`nussl.AudioSignal` object is the main container for all things related to your audio data. It provides a lot of
helpful utilities to make it easy to manipulate your audio. Because it is at the heart of all of the source separation
algorithms in *nussl*, it is crucial to understand how it works. Here we provide a brief introduction to many common
tasks.

Initialization from a file
--------------------------

It is easy to initialize an AudioSignal object by loading an audio file from a path

>>> from __future__ import division
>>> import nussl
>>> input_file_path = 'path/to/input.wav'
>>> signal1 = nussl.AudioSignal(input_file_path)

Now the AudioSignal object is ready with all of the information about our the signal.

>>> print("{} Hz".format(signal1.sample_rate))  # This is determined by our input file
44100 Hz
>>> signal1.num_channels
2
>>> print("{} seconds".format(signal1.signal_duration))
2 seconds
>>> print("{} samples".format(signal1.signal_length))  # Two seconds at 44.1 kHz means...
88200 samples
>>> signal1.file_name  # just get the file name
'input.wav'
>>> signal1.path_to_input_file  # this is the full path
'path/to/input.wav'

The actual signal data is in ``signal1.audio_data``. It’s just a numpy array, so we can use it as such:

>>> signal1.audio_data
array([[  0.00000000e+00,   0.00000000e+00,   6.10351562e-05, ...,
          2.47192383e-03,   1.04370117e-02,   1.83410645e-02],
       [  0.00000000e+00,   0.00000000e+00,   6.10351562e-05, ...,
          2.47192383e-03,   1.04370117e-02,   1.83410645e-02]], dtype=float32)
>>> signal1.audio_data.shape
(2, 88200)

A few things to note here:

1. When AudioSignal loads a file, it converts the data to floats between [-1, 1]
2. The number of channels is the first dimension, the number of samples is the second.

Initialization from a numpy array
---------------------------------

Another common way to initialize an AudioSignal object is by passing in a numpy array. Let’s first
make a single channel signal within a numpy array.

>>> sample_rate = 44100  # Hz
>>> dt = 1.0 / sample_rate
>>> dur = 2.0  # seconds
>>> freq = 5000  # Hz
>>> x = np.arange(0.0, dur, dt)
>>> x = np.sin(2 * np.pi * freq * x)

Cool! Now let’s put this into a new AudioSignal object.

>>> signal2 = nussl.AudioSignal(audio_data_array=x)
>>> len(x) == signal2.signal_length == len(signal2)  # These are all the same thing, right?
True
>>> signal2.rms()  # Root-mean-square of audio_data in signal2
0.70710678118654524
>>> signal2.time_vector  # This is a vector with timestamps at each sample
array([  0.00000000e+00,   2.26759941e-05,   4.53519881e-05, ...,
         1.99995465e+00,   1.99997732e+00,   2.00000000e+00])
>>> signal2.audio_data_as_ints()  # Get signal2.audio_data as ints
array([[     0,  21418,  32419, ..., -27651, -32419, -21418]], dtype=int16)

Alright, alright, alright [#f1]_.

Other basic manipulations
-------------------------

If we want to add the audio data in these two signals, it's simple. But there are some gotchas:

>>> signal3 = signal1 + signal2
Exception: Cannot add with two signals that have a different number of channels!

Uh oh! I guess it doesn’t make sense to add a stereo signal (``signal1``) and mono signal (``signal2``).
But if we really want to add these two signals, we have a few options.

First, we can just get one of the channels like this:

>>> signal1.get_channel(0)

Another option we have is to we can make ``signal1`` mono. *nussl* does this by simply averaging the
two channels at every sample. We have to explicitly tell *nussl* that we are okay with ``to_mono()``
changing ``audio_data``. We do that like this:

>>> signal1.to_mono(overwrite=True)

If we hadn’t set ``overwrite=True`` then ``to_mono()`` would just return a np array of mono-ed ``audio_data`` and
not change the representation of ``signal1.audio_data``. You will see this pattern come up again. In certain
places, :class:`AudioSignal`:'s default behavior is to overwrite its internal data, and in other places the default is to
**not** overwrite data. See the reference pages for more info.

Now, we can add the two signals as before:

>>> signal3 = signal1 + signal2

No exceptions this time! Great! ``signal3`` is now a new :class:`AudioSignal`: object. We can similarly subtract two signals.

Let’s write this to a file:

>>> signal3.write_audio_to_file('path/to/output.wav')

Awesome! Now lets see how we can manipulate the audio in the frequency domain...

.. rubric:: Footnotes

.. [#f1] Here, ``signal2`` has no value for ``file_name`` or ``path_to_input_file``. They are ``None``.
