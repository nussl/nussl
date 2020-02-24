.. _audio_signal_stft:

======================
Spectrograms and STFTs
======================

Most, if not all, source separation algorithms do not their operations in the time domain, but rather in the frequency
domain. For this, *nussl* provides an interface for working with `Short-Time Fourier Transform (STFT) <https://en.wikipedia.org/wiki/Short-time_Fourier_transform>`_
data. Here, we describe how to do some simple STFT operations with the :class:`nussl.AudioSignal` object.

STFT Basics
-----------

Let's reinitialize ``signal1`` from the previous page. We should be able to get frequency domain data by looking
at ``signal1.stft_data``. Let's try that.

>>> signal1 = nussl.AudioSignal(input_file_path)
>>> signal1.stft_data
None

Whoops! Because this object was initialized from a .wav file (i.e., time-series data), this :class:`AudioSignal`: object has
no frequency domain data by default. To populate it with frequency data we do thusly:

>>> signal1.stft()

Aha! Now we can examine how STFT data is stored in the :class:`AudioSignal`: object. Similar to ``signal1.audio_data``,
STFT data is stored in a (complex-valued) numpy array called ``signal1.stft_data`` [#f1]_.

>>> signal1.stft_data
array([[[  5.65585184e+00 -0.00000000e+00j,
           9.01010437e+01 -0.00000000e+00j],
        [ -2.49999994e-03 -3.83490305e-06j,
          -2.49999994e-03 -3.83490305e-06j],
        [ -2.49999994e-03 -3.83490305e-06j,
          -2.49999994e-03 -3.83490305e-06j],
        [ -2.49999994e-03 -3.83490305e-06j,
          -2.49999994e-03 -3.83490305e-06j],
        [ -4.15182253e-03 -3.05598299e-03j,
          -5.95030794e-03 -6.10865979e-03j],
        [  7.39212409e-02 +2.67153326e-02j,
           1.48953497e-01 +5.34350201e-02j]],
       ...,
      [[ -1.25701912e-03 -3.83491215e-06j,
           6.89334124e-02 +2.11507810e-04j],
        [ -2.49999994e-03 -7.66982430e-06j,
          -2.49999994e-03 -7.66982430e-06j],
        [ -2.49999994e-03 -7.66982430e-06j,
          -2.49999994e-03 -7.66982430e-06j],
        [ -2.49999994e-03 -7.66982430e-06j,
          -2.49999994e-03 -7.66982430e-06j],
        [ -2.51177116e-03 -5.69828972e-03j,
          -2.67004268e-03 -1.13899261e-02j],
        [ -4.96160686e-02 -4.97913919e-02j,
          -1.36735573e-01 -0.00000000e+00j]]], dtype=complex64)
>>> signal1.stft_data.shape
(1025, 88, 2)

By inspecting the shape we see that the first dimension represents the number of FFT bins taken at each hop,
the second represents the length of our signal (in hops), and the third dimension is number of channels. There is
an easy way to get all of this data from our :class:`AudioSignal`: object without having to do array indexing:

>>> signal1.stft_length
88
>>> signal1.num_fft_bins
1025

We can get a single STFT channel like so:

>>> signal1.get_stft_channel(1)  # see footnote 2 on AudioSignal Basics page
array([[ 1.23336256 -0.00000000e+00j,  0.03598116 -0.00000000e+00j,
         0.10520950 -0.00000000e+00j, ...,  0.06182364 -0.00000000e+00j,
        -1.39272857 -0.00000000e+00j, -0.27395117 -0.00000000e+00j],
       [-1.45443594 +2.95939099e-04j,  0.03598259 +4.79334674e-04j,
         0.10521182 +1.84503690e-04j, ...,  0.06182393 -5.87543298e-04j,
         1.07824659 +5.23063958e-01j, -0.26018760 -6.48025423e-02j],
       [ 1.23345983 +5.91909746e-04j,  0.03598688 +9.58720455e-04j,
         0.10521877 +3.69027053e-04j, ...,  0.06182481 -1.17514888e-03j,
        -0.96882135 -9.56665993e-01j, -0.22125557 -1.18454359e-01j],
       ...,
       [-0.19744445 -8.19548659e-05j, -0.04824998 -1.32742891e-04j,
        -0.03122028 -5.10948921e-05j, ...,  0.04447741 +1.62709228e-04j,
         0.13960634 -1.32384062e-01j,  0.00856382 -1.65341552e-02j],
       [ 0.17461725 -4.09772983e-05j, -0.04824989 -6.63712271e-05j,
        -0.03122014 -2.55473606e-05j, ...,  0.04447743 +8.13543447e-05j,
        -0.14375341 +7.24206716e-02j,  0.01395597 -9.03958548e-03j],
       [-0.19744259 -0.00000000e+00j, -0.04824987 -0.00000000e+00j,
        -0.03122010 -0.00000000e+00j, ...,  0.04447743 -0.00000000e+00j,
         0.19829373 -0.00000000e+00j,  0.01586237 -0.00000000e+00j]], dtype=complex64)

We can also get power spectrogram data from our signal as well. As we would expect, this is the same
shape as ``signal1.stft_data``.

>>> signal1.power_spectrogram_data  # np.abs(signal1.stft_data) ** 2
array([[[  1.52118325e+00],
        [  1.29464362e-03],
        [  1.10690389e-02],
        ...,
        [  1.97824207e-03],
        [  3.93204018e-02],
        [  2.51614663e-04]]], dtype=float32)
>>> signal1.power_spectrogram_data.shape
(1025, 88, 2)

Inverse STFTs
-------------

Let's do something a little more interesting with our :class:`AudioSignal`: object. Since ``signal1.stft_data`` is just
a regular numpy array, we can access and manipulate it as such. So let's implement a low pass filter by creating a
new :class:`AudioSignal`: object and leaving ``signal1`` unaltered.

Let's eliminate all frequencies above about 400 Hz in our signal.

>>> lp_stft = signal1.stft_data.copy()
>>> lp_cutoff = 400  # Hz
>>> frequency_vector = signal1.freq_vector  # a vector of frequency values for each FFT bin
>>> idx = (np.abs(frequency_vector - lp_cutoff)).argmin()  # trick to find the index of the closest value to 400 Hz
>>> lp_stft[idx:, :, :] = 0.0j  # every freq above ~400 Hz is 0 now

Okay, so now we have low passed STFT data in the numpy array ``lp_stft``. Now we are going to see how we can initialize
a new :class:`AudioSignal`: object using this data.

>>> signal1_lp = nussl.AudioSignal(stft=lp_stft)

Easy-peasy! Now ``signal1_lp`` is a new :class:`AudioSignal`: object that has been initialized with STFT data instead of
time series data. Before we can write this to a file, we need to do an Inverse STFT to get back time-series data.

>>> signal1_lp.audio_data
None
>>> signal1_lp.istft()
>>> signal1_lp.write_audio_to_file('path/to/signal1_lowpass.wav')

Cool beans!


STFT Parameters
---------------

I wanted to make a few quick notes about *nussl*'s parameter settings for STFTs and iSTFTs. Let's have a quick look
at the function signature for ``AudioSignal.stft()`` and ``AudioSignal.istft()``:

.. code-block:: python

    def stft(self, window_length=None, hop_length=None, window_type=None, n_fft_bins=None,
             remove_reflection=True, overwrite=True, use_librosa=True):
        ...

    def istft(self, window_length=None, hop_length=None, window_type=None, n_fft_bins=None, overwrite=True,
              reconstruct_reflection=True, use_librosa=True):
        ...

Almost all low level parameters are accessible through this interface and can be adjusted accordingly.

As with ``to_mono()`` on the previous page, ``stft()`` and ``istft()`` have parameters to overwrite the internal
data representations. By default they are true, be sure to set them to false when needed.

While *nussl* does have its own STFT and iSTFT implementations, it also contains wrappers for
`librosa's <https://librosa.github.io/librosa/generated/librosa.core.stft.html#librosa.core.stft>`_ STFT and iSTFT
functions. There is a trade off to both: based on our tests *librosa's* is faster, but *nussl's* produces more accurate signal
reconstruction. Some algorithms produce artifacts with *nussl's* STFTs, so *nussl* defaults to using librosa STFT functions [#f2]_.

The default
settings for forward and inverse STFTs are guaranteed to produce invertible results without crashing. But because
there are so many possibilities, *nussl* assumes the user will know what the correct way to compute both STFT and
iSTFT correctly. E.g., if you do not remove the FFT reflection when doing an STFT, *nussl* will not automatically
know not to reconstruct the reflection when doing an inverse STFT. It is the user's responsibility to do this
kind of bookkeeping.

StftParams Object
^^^^^^^^^^^^^^^^^

The :class:`nussl.StftParams` is an object that stores parameters for doing STFTs and iSTFTs. Its main function is to
keep track of user-set parameters for the duration of the :class:`AudioSignal`: object's life. The separation
objects (:class:`nussl.Repet`, for instance) also have a ``StftParams`` object, which is copied from the input
:class:`AudioSignal`: object.

Let's get to some examples to demonstrate.

We can calculate an STFT with special, non-default parameters:

>>> signal1 = nussl.AudioSignal(input_file_path)
>>> signal1.stft(window_length=4096, hop_length=1024, window_type=constants.WINDOW_HANN)

But the next time we do an STFT, we'll have the default parameters again. And *furthermore*, any other code
that calls ``signal1.sift()`` (like, ``nussl.Repet``) will use the default parameters.

If we want to have these settings saved for the life of this :class:`AudioSignal`: object, we have to set them
in ``signal1``'s ``StftParams`` object, at ``signal1.stft_params``.

>>> signal1 = nussl.AudioSignal(input_file_path)
>>> signal1.stft_params.window_length = 4096
>>> signal1.stft_params.hop_length = 1024
>>> signal1.stft_params.window_type = constants.WINDOW_HANN
>>> signal1.stft()

This block of code is equivalent to the first, but now when we call ``signal1.stft()`` again it will use the same
settings again.

So now when ``Repet`` does an STFT (or any other separation method in *nussl*), it will use our settings again.

>>> my_repet = nussl.Repet(signal1)
>>> my_repet.run()

The STFT inside ``Repet`` used the parameters we set for ``signal1``. Automatically, ``my_repet`` has a copy of
``signal1.stft_params``:

>>> my_repet.stft_params.window_length
4096
>>> my_repet.stft_params.hop_length
1024
>>> my_repet.stft_params.window_type
'hann'

What if I want to change just one of the STFT parameters for only this repet instance? We can change the STFT parameters
for ``my_repet`` and they won't affect ``signal1.stft_params``.

>>> my_repet.stft_params.hop_length == 2048
>>> my_repet.run()

Now when we run ``my_repet``, the hop length is 2048 instead of what was set in ``signal1``, 1024.

`Awesome! <http://i.giphy.com/d2Z9QYzA2aidiWn6.gif>`_

.. rubric:: Footnotes

.. [#f1] All of the python console output on this page has been truncated for brevity.
.. [#f2] This may change in a future release.