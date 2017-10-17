#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo of NMF with MFCC clustering in nussl
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
    A simple example of Non-Negative Matrix Factorization (NMF) with k-Means MFCC clustering in nussl.
    See nussl documentation for more information about using NMF_MFCC.
    Returns:

    """
    # Load input file
    input_file_name = os.path.join('..', 'input', 'piano_and_synth_arp_chord_mono.wav')
    signal = nussl.AudioSignal(path_to_input_file=input_file_name)

    # make a directory to store output if needed
    if not os.path.exists(os.path.join('..', 'Output/')):
        os.mkdir(os.path.join('..', 'Output/'))

    # Set up NMMF MFCC
    nmf_mfcc = nussl.NMF_MFCC(signal, num_sources=2, num_templates=6, num_iterations=10, random_seed=0,
                              distance_measure=nussl.TransformerNMF.EUCLIDEAN)
    # and run
    nmf_mfcc.run()
    sources = nmf_mfcc.make_audio_signals()
    for i, source in enumerate(sources):
        output_file_name = '{}_{}.wav'.format(os.path.splitext(signal.file_name)[0], i)
        source.write_audio_to_file(output_file_name)


if __name__ == '__main__':
    main()
