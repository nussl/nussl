#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
import unittest
import nussl


class BenchmarkFileLoadError(Exception):
    """
    Exception class for when a downloaded benchmark file fails to load.
    """
    pass


class BenchmarkTestBase(unittest.TestCase):
    """
    This is a base class for running a benchmark tests in nussl. This is a minimal class that
    has resources for loading and removing benchmark files from the nussl External File Zoo (EFZ).
    """

    @classmethod
    def load_benchmarks(cls, assert_type=None):
        """
        This function will download all of the required benchmark files from
        the nussl External File Zoo (EFZ) and load them into a dictionary. The returned
        dictionary will have the

        Args:
            assert_type (None): If not `None`, this function expects a string with an
            extension type. If given an extension type, this function will reject
        Returns:

        """
        benchmark_dict = {}
        benchmark_metadata_list = nussl.efz_utils.get_available_benchmark_files()
        for benchmark_file in benchmark_metadata_list:
            if benchmark_file['for_class'] == cls.__name__:
                name, ext = os.path.splitext(benchmark_file['file_name'])
                if assert_type and ext == assert_type:
                    file_path = nussl.efz_utils.download_benchmark_file(benchmark_file['file_name'])

                    benchmark = np.load(file_path)
                    benchmark_dict[name] = benchmark

        return benchmark_dict

    @staticmethod
    def _load_file(file_path):
        """
        This function actually loads the benchmark file. At the time of this writing, it only
        supports numpy (.npy) files.
        Args:
            file_path (string):

        Returns:

        """
        name, ext = os.path.splitext(os.path.basename(file_path))

        if ext == 'npy':
            return np.load(file_path)
        elif ext in ('mat', 'm'):
            # We need more information
            os.remove(file_path)
            raise BenchmarkFileLoadError('Cannot load MATLAB file at this time.')
        else:
            os.remove(file_path)
            raise BenchmarkFileLoadError('Unknown file type {}.'.format(ext))

    @classmethod
    def remove_benchmarks(cls):
        """

        Returns:

        """
        benchmark_metadata_list = nussl.efz_utils.get_available_benchmark_files()
        for benchmark_file in benchmark_metadata_list:
            if benchmark_file['for_class'] == cls.__name__:

                # This is a lil hack: download_benchmark_file() will give us the path
                # to the file (because it's already been downloaded)
                file_path = nussl.efz_utils.download_benchmark_file(benchmark_file['file_name'],
                                                                    verbose=False)

                os.remove(file_path)
