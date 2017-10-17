#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo for the HighLowPassFilter class
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
    # Load an Audio file
    file_name = 'dev1_wdrums_inst_mix.wav'
    input_path = os.path.abspath(os.path.join('input', file_name))
    signal = nussl.AudioSignal(input_path)

    # Make an output path
    output_folder = os.path.join('Output', 'high_low_pass_output')

    #
    high_pass_cutoff = 2000  # Hz

    hlpf = nussl.separation.HighLowPassFilter(signal, high_pass_cutoff, do_fir_filter=True)
    high_pass_mask, low_pass_mask = hlpf.run()
    high_pass_signal, low_pass_signal = hlpf.make_audio_signals()

    high_pass_signal.write_audio_to_file(os.path.join(output_folder, 'high_pass_fir.wav'))
    high_pass_signal.plot_spectrogram(os.path.join(output_folder, 'high_pass_fir.png'))
    low_pass_signal.write_audio_to_file(os.path.join(output_folder, 'low_pass_fir.wav'))
    low_pass_signal.plot_spectrogram(os.path.join(output_folder, 'low_pass_fir.png'))


    hlpf = nussl.separation.HighLowPassFilter(signal, high_pass_cutoff, do_fir_filter=False)
    high_pass_mask, low_pass_mask = hlpf.run()
    high_pass_signal, low_pass_signal = hlpf.make_audio_signals()

    high_pass_signal.write_audio_to_file(os.path.join(output_folder, 'high_pass_mask.wav'))
    high_pass_signal.plot_spectrogram(os.path.join(output_folder, 'high_pass_mask.png'))
    low_pass_signal.write_audio_to_file(os.path.join(output_folder, 'low_pass_mask.wav'))
    low_pass_signal.plot_spectrogram(os.path.join(output_folder, 'low_pass_mask.png'))


if __name__ == '__main__':
    main()
