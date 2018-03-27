#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utils for interfacing nussl with common data sets
"""
import os
import hashlib
import warnings

import numpy as np

import utils
from audio_signal import AudioSignal
import constants

__all__ = ['iKala', 'mir1k']


def _hash_directory(directory, ext=None):

    hash_list = []
    for path, sub_dirs, files in os.walk(directory):
        if ext is None:
            hash_list.extend([utils._hash_file(os.path.join(path, f)) for f in files
                              if os.path.isfile(os.path.join(path, f))])
        else:
            hash_list.extend([utils._hash_file(os.path.join(path, f)) for f in files
                              if os.path.isfile(os.path.join(path, f))
                              if os.path.splitext(f)[1] == ext])

    hasher = hashlib.sha256()
    for hash_val in sorted(hash_list):
        hasher.update(hash_val.encode('utf-8'))

    return hasher.hexdigest()


def _data_set_setup(directory, top_dir_name, audio_dir_name, expected_hash, check_hash, ext):

    # Verify the top-level directory is correct
    if not os.path.isdir(directory):
        raise DataSetException('Expected directory, got \'{}\'!'.format(directory))

    if top_dir_name != os.path.split(directory)[1]:
        raise DataSetException('Expected {}, got \'{}\''.format(top_dir_name,
                                                                os.path.split(directory)[1]))

    # This should be tha directory with all of the audio files
    audio_dir = os.path.join(directory, audio_dir_name)
    if not os.path.isdir(audio_dir):
        raise DataSetException('Expected {} to be a directory but it is not!'.format(audio_dir))

    # Check to see if the contents are what we expect
    # by hashing all of the files and checking against a precomputed expected_hash
    if _hash_directory(audio_dir, ext) != expected_hash:
        msg = 'Hash of {} does not match known directory hash!'.format(audio_dir)
        if check_hash:
            raise DataSetException(msg)
        else:
            warnings.warn(msg)

    # return a list of the full paths of every audio file
    return [os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
            if os.path.isfile(os.path.join(audio_dir, f))
            if os.path.splitext(f)[1] == ext]


def _subset_and_shuffle(file_list, subset, shuffle, seed):

    if shuffle:
        if seed is not None:
            np.random.seed(seed)

        np.random.shuffle(file_list)

    if isinstance(subset, float):
        if subset < 0.0 or subset > 1.0:
            raise DataSetException('subset must be a number between [0.0, 1.0]!')

        result = file_list[:int(subset*len(file_list))]

    elif isinstance(subset, list):
        result = [file_list[i] for i in subset]

    elif subset is None:
        result = file_list

    else:
        raise DataSetException('subset must be a list of indices or float between [0.0, 1.0]!')

    return result


def iKala(directory, check_hash=True, subset=None, shuffle=False, seed=None):
    """
    Generator function for the iKala data set.
    Args:
        directory:
        check_hash:
        subset:
        shuffle:
        seed:

    Returns:

    """

    top_dir_name = 'iKala'
    audio_dir_name = 'Wavfile'
    iKala_hash = 'd82191c73c3ce0ab0ed3ca21a3912769394e8c9a16d742c6cc446e3f04a9cd9e'
    audio_extension = '.wav'
    all_wav_files = _data_set_setup(directory, top_dir_name, audio_dir_name,
                                    iKala_hash, check_hash, audio_extension)

    all_wav_files = _subset_and_shuffle(all_wav_files, subset, shuffle, seed)

    for f in all_wav_files:
        mixture = AudioSignal(f)
        singing = mixture.make_audio_signal_from_channel(1)
        accompaniment = mixture.make_audio_signal_from_channel(0)

        yield mixture, singing, accompaniment


def mir1k(directory, check_hash=True, subset=None, shuffle=False, seed=None, undivided=False):
    """
    Generator function for the MIR-1K data set.
    Args:
        directory:
        check_hash:
        subset:
        shuffle:
        seed:
        undivided:

    Returns:

    """

    top_dir_name = 'MIR-1K'

    wavfile_hash = '33c085c1a7028199cd20317868849b413e0971022ebc4aefcf1bbc5516646c29'
    undivided_hash = '3f39af9be17515e042a7005b4c47110c6738623a7fada6233ba104535b7dde1b'

    if undivided:
        audio_dir_name = 'UndividedWavfile'
        mir1k_hash = undivided_hash
    else:
        audio_dir_name = 'Wavfile'
        mir1k_hash = wavfile_hash

    audio_extension = '.wav'
    all_wav_files = _data_set_setup(directory, top_dir_name, audio_dir_name,
                                    mir1k_hash, check_hash, audio_extension)

    all_wav_files = _subset_and_shuffle(all_wav_files, subset, shuffle, seed)

    for f in all_wav_files:
        mixture = AudioSignal(f)
        singing = mixture.make_audio_signal_from_channel(1)
        accompaniment = mixture.make_audio_signal_from_channel(0)

        yield mixture, singing, accompaniment


class DataSetException(Exception):
    pass
