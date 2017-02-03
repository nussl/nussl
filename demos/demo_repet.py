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
    input_name = os.path.join('..', 'input','mix1.wav')
    signal = nussl.AudioSignal(path_to_input_file=input_name)
    original_length = signal.signal_length
    signal.stft_params.window_length = 2048
    signal.stft_params.n_fft_bins = 2048
    signal.stft_params.hop_length = 1024
    original_audio = np.copy(signal.audio_data)
    signal.stft() 
    print signal.stft().shape
    signal.istft()
    print signal.signal_length
    print np.sum(np.abs(original_audio - signal.audio_data))
    # make a directory to store output if needed
    if not os.path.exists('output/'):
        os.mkdir('output/')

    # Set up Repet
    repet = nussl.Repet(signal)

    # and Run
    repet.run()

    # Get foreground and backgroun audio signals
    bkgd, fgnd = repet.make_audio_signals()
    print original_length, bkgd.signal_length, fgnd.signal_length
    # and write out to files
    bkgd.write_audio_to_file(os.path.join('output', 'mix1_bg.wav'))
    fgnd.write_audio_to_file(os.path.join('output', 'mix1_fg.wav'))


if __name__ == '__main__':
    main()
