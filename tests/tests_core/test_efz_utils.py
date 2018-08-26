#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for nussl/core/efz_utils.py
"""

import unittest
import nussl
import os


class TestEfzUtils(unittest.TestCase):

    @staticmethod
    def _remove_file(target_file, verbose=True):
        base_path = os.path.expanduser(os.path.join('~', '.nussl'))
        if not os.access(base_path, os.W_OK):
            base_path = os.path.join('/tmp', '.nussl')

        for root, _, file_names in os.walk(base_path):
            for cur_file in file_names:
                if cur_file == target_file:
                    # noinspection PyBroadException
                    try:
                        found_file = os.path.join(root, target_file)
                        os.remove(found_file)
                        assert not os.path.exists(found_file)
                        if verbose:
                            print('Removed {}.'.format(found_file))
                    except Exception:
                        pass

    def test_download_simple(self):
        files = {nussl.efz_utils.download_audio_file: 'K0140.wav',
                 nussl.efz_utils.download_trained_model: 'deep_clustering_vocals_44k_long.model',
                 }

        for func, file_ in files.items():
            self._remove_file(file_)
            path = func(file_)
            assert os.path.isfile(path)
            os.remove(path)

    def test_print_available_trained_models(self):
        nussl.efz_utils.print_available_trained_models()

    def test_print_available_audio(self):
        nussl.efz_utils.print_available_audio_files()

    def test_print_available_benchmark_files(self):
        nussl.efz_utils.print_available_benchmark_files()
