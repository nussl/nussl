import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)

import nussl

def main():
    path_to_file1 = '../Input/src1.wav'
    path_to_file2 = '../Input/src2.wav'

    # Load the file into the AudioSignal object
    signal1 = nussl.AudioSignal(path_to_file1)
    stft = signal1.stft(remove_reflection=False)
    signal1.istft()

    n_ch = signal1.num_channels
    signal1.write_audio_to_file('../Output/duet_test1.wav')

if __name__ == '__main__':
    main()