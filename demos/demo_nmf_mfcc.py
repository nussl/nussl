#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import nussl
import numpy as np

def main():
    # Load input file


    input_file_name = os.path.join('..', 'input', 'dev1_female3_inst_mix.wav')
    signal = nussl.AudioSignal(path_to_input_file=input_file_name)

    # Set up DUET
    nmf_mfcc = nussl.NMF_MFCC(signal, num_sources= 3, num_templates=60, distance_measure="euclidean", num_iterations=50)

    # and run
    nmf_mfcc.run()
    nmf_mfcc.make_audio_signals()

if __name__ == '__main__':
    main()
