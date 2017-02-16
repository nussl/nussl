import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)

import nussl
print nussl.USE_LIBROSA_STFT
import numpy as np

def main():
    # input audio file
    input_name = os.path.join('..', 'input','demo_mixture.wav')
    signal = nussl.AudioSignal(path_to_input_file=input_name)
    #signal.to_mono(overwrite = True)
    original_length = signal.signal_length
    signal.stft_params.window_length = 2048
    signal.stft_params.n_fft_bins = 2048
    signal.stft_params.hop_length = 512
    print signal.num_channels
    # make a directory to store output if needed
    if not os.path.exists('output/'):
        os.mkdir('output/')

    # Set up Repet
    ft2d = nussl.FT2D(signal)

    # and Run
    ft2d.run()

    # Get foreground and backgroun audio signals
    bkgd, fgnd = ft2d.make_audio_signals()
    print original_length, signal.signal_length, bkgd.signal_length, fgnd.signal_length
    # and write out to files
    bkgd.write_audio_to_file(os.path.join('output', 'mix1_bg.wav'))
    fgnd.write_audio_to_file(os.path.join('output', 'mix1_fg.wav'))
    print bkgd._active_start, fgnd._active_start
    print bkgd._active_end, fgnd._active_end
    audio_sum =  bkgd + fgnd
    audio_sum.write_audio_to_file(os.path.join('output', 'sum.wav'))

if __name__ == '__main__':
    main()
