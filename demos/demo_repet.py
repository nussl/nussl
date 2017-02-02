import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)

import nussl


def main():
    # input audio file
    input_name = os.path.join('..', 'Input','guitar2__cowbell1_none_0.0.wav')
    signal = nussl.AudioSignal(path_to_input_file=input_name)

    # make a directory to store output if needed
    if not os.path.exists('Output/'):
        os.mkdir('Output/')

    # Set up Repet
    repet = nussl.Repet(signal)

    # and Run
    repet.run()

    # Get foreground and backgroun audio signals
    bkgd, fgnd = repet.make_audio_signals()

    # and write out to files
    bkgd.write_audio_to_file(os.path.join('..', 'Output', 'cowbell1_guitar_background_lib2.wav'))
    fgnd.write_audio_to_file(os.path.join('..', 'Output', 'cowbell1_guitar_foreground_lib2.wav'))


if __name__ == '__main__':
    main()
