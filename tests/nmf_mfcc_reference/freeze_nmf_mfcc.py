#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import platform

import numpy as np

import nussl

DEBUG = False


def main():
    """
    Freezes essential values from NMF MFCC in its current implementation for benchmarking
    See test_benchmark_nmf_mfcc() in test_nmf_mfcc.py for usage.

    Run with top level ('nussl/') as working directory

    """

    metadata = {}

    path_to_input_file = os.path.join('input', 'piano_and_synth_arp_chord_mono.wav')
    metadata['input_file'] = path_to_input_file
    signal = nussl.AudioSignal(path_to_input_file)

    # Set random seed in NMF and KMeans to 0
    params = {'num_sources': 2, 'num_templates': 6, 'distance_measure': nussl.transformers.TransformerNMF.EUCLIDEAN,
              'num_iterations': 10, 'random_seed': 0}
    metadata['params'] = params

    nmf_mfcc = nussl.NMF_MFCC(signal, **params)

    if DEBUG:
        output_folder = os.path.join('tests', 'nmf_mfcc_reference', 'scratch')
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
    else:
        output_folder = os.path.join('tests', 'nmf_mfcc_reference', 'nmf_mfcc_benchmark_files')

    nmf_mfcc.run()
    np.save(os.path.join(output_folder, 'benchmark_labeled_templates'), nmf_mfcc.labeled_templates)
    np.save(os.path.join(output_folder, 'benchmark_masks'), nmf_mfcc.result_masks)

    nmf_mfcc.make_audio_signals()

    # Make sure the paths are empty
    for source in nmf_mfcc.sources:
        source.path_to_input_file = ''

    np.save(os.path.join(output_folder, 'benchmark_sources'), nmf_mfcc.sources)
    metadata['save_time'] = time.asctime()
    metadata['nussl_version'] = nussl.version
    metadata['made_by'] = platform.uname()[1]
    json.dump(metadata, open(os.path.join(output_folder, 'nmf_mfcc_benchmark_metadata.json'), 'w'))


if __name__ == '__main__':
    main()
