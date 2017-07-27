#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo for the HighLowPassFilter class
"""

import os
import nussl

def main():
    path = os.path.abspath(os.path.join('input', 'dev1_wdrums_inst_mix.wav'))
    signal = nussl.AudioSignal(path)

    high_pass_cutoff = 200  # Hz

    hlpf = nussl.separation.HighLowPassFilter(signal, high_pass_cutoff, do_fir_filter=True)
    high_pass_mask, low_pass_mask = hlpf.run()
    high_pass_signal, low_pass_signal = hlpf.make_audio_signals()

    hlpf = nussl.separation.HighLowPassFilter(signal, high_pass_cutoff, do_fir_filter=False)
    high_pass_mask, low_pass_mask = hlpf.run()
    high_pass_signal, low_pass_signal = hlpf.make_audio_signals()


if __name__ == '__main__':
    main()
