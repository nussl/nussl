#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

try:
    # import from an already installed version
    import nussl
except:

    # can't find an installed version, import from right next door...
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not path in sys.path:
        sys.path.insert(1, path)

    import nussl


def main():
    """
    Basics of the AudioSignal object in nussl. See the articles at
    https://interactiveaudiolab.github.io/nussl/getting_started/audio_signal_basics.html
    for a more detailed walk-through.
    Returns:

    """
    path_to_file1 = os.path.join('input', 'src1.wav')

    # Load the first file into the AudioSignal object
    signal1 = nussl.AudioSignal(path_to_file1)

    # Now there's also a bunch of metadata, we can check out
    print("Sample Rate: {} Hz".format(signal1.sample_rate))
    print("Number of channels: {}".format(signal1.num_channels))
    print("Length: {} seconds".format(signal1.signal_duration))
    print("Length: {} samples".format(signal1.signal_length))
    print("File name: {}".format(signal1.file_name))
    print("Path to file: {}".format(signal1.path_to_input_file))

    # Time series data is in the .audio_data attribute, as a 2D numpy array
    print("signal1.audio_data.shape = {}".format(signal1.audio_data.shape))

    # It's easy to do an Short-Time Fourier Transform (STFT)
    signal1.stft()

    # The STFT data is stored here as a 3D complex numpy array
    print("signal1.stft_data.shape = {}".format(signal1.stft_data.shape))

    # After you're done manipulating the STFT data, you can go back to the time domain like this...
    signal1.istft()

    # Now let's load another file
    path_to_file2 = os.path.join('input', 'src2.wav')
    signal2 = nussl.AudioSignal(path_to_file2)

    # Let's mix these two signals, but before we do:
    # signal2 is longer than signal1, so let's truncate signal2
    signal2.truncate_samples(signal1.signal_length)

    # Now we can create a simple mixture like so
    signal3 = signal1.apply_gain(0.5) + signal2.apply_gain(0.3)

    # And write it to a file
    signal3.write_audio_to_file(os.path.join('Output', 'mixture.wav'))


if __name__ == '__main__':
    main()
