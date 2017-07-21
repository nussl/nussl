import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)

import nussl
import numpy as np


def repet_test():
    path = os.path.join('..', 'Input', 'mix3.wav')
    signal = nussl.AudioSignal(path)

    ft = nussl.FT2D(signal, use_librosa_stft=False)
    ft()

    back, fore = ft.make_audio_signals()

    recombined = back + fore
    recombined.write_audio_to_file(os.path.join('..', 'Output', 'mix3_recombined_ft2d_nussl.wav'))

if __name__ == '__main__':
    repet_test()