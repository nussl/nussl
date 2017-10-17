#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo for 2DFT (called FT2D in nussl) separation
"""

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
    Demo for 2DFT (called FT2D in nussl) separation
    Returns:

    """
    # input audio file
    input_name = os.path.join('..', 'input','demo_mixture.wav')
    signal = nussl.AudioSignal(path_to_input_file=input_name)

    signal.stft_params.window_length = 2048
    signal.stft_params.n_fft_bins = 2048
    signal.stft_params.hop_length = 512

    # make a directory to store output if needed
    if not os.path.exists('output/'):
        os.mkdir('output/')

    # Set up FT2D
    ft2d = nussl.FT2D(signal)

    # and Run
    ft2d.run()

    # Get foreground and backgroun audio signals
    bkgd, fgnd = ft2d.make_audio_signals()

    # and write out to files
    bkgd.write_audio_to_file(os.path.join('output', 'mix1_bg.wav'))
    fgnd.write_audio_to_file(os.path.join('output', 'mix1_fg.wav'))

    audio_sum =  bkgd + fgnd
    audio_sum.write_audio_to_file(os.path.join('output', 'sum.wav'))


if __name__ == '__main__':
    main()
