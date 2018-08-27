#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Here are the utilities for using the nussl External File Zoo (EFZ).
"""

import warnings
import json
import os
import sys
import hashlib

from six.moves.urllib_parse import urljoin
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlopen, Request

import constants


__all__ = ['get_available_audio_files', 'print_available_audio_files',
           'get_available_benchmark_files', 'print_available_benchmark_files',
           'get_available_trained_models', 'print_available_trained_models',
           'download_audio_file', 'download_benchmark_file', 'download_trained_model',
           'FailedDownloadError', 'MismatchedHashError', 'MetadataError']


def get_available_audio_files():
    """
    Returns a dict containing metadata of the available files on the nussl External File Zoo
    (EFZ) server (http://nussl.ci.northwestern.edu/).
    Returns:
        (dict): containing metadata of the available files on the nussl External File Zoo
            (EFZ) server (http://nussl.ci.northwestern.edu/).

    """
    # _download_metadata_file() will throw its own errors, so no try block around it
    return _download_metadata_file(constants.NUSSL_EFZ_AUDIO_METADATA_URL)


def print_available_audio_files():
    """
    Prints a message to the console that shows all of the available audio files that are on the
    nussl External File Zoo (EFZ) server (http://nussl.ci.northwestern.edu/).

    Example:
        >>> import nussl
        >>> nussl.efz_utils.print_available_audio_files()

        File Name                                Duration (sec)  Size       Description
        dev1_female3_inst_mix.wav                10.0            1.7MiB     Instantaneous mixture of three female speakers talking in a stereo field.
        dev1_female3_synthconv_130ms_5cm_mix.wav 10.0            1.7MiB     Three female speakers talking in a stereo field, with 130ms of inter-channel delay.
        K0140.wav                                5.0             431.0KiB   Acoustic piano playing middle C.
        K0149.wav                                5.0             430.0KiB   Acoustic piano playing the A above middle C. (A440)

        Last updated 2018-03-06

    """
    file_metadata = get_available_audio_files()

    print('{:40} {:15} {:10} {:50}'.format('File Name', 'Duration (sec)',
                                           'Size', 'Description'))
    for f in file_metadata:
        print('{:40} {:<15.1f} {:10} {:50}'.format(f['file_name'], f['file_length_seconds'],
                                                   f['file_size'], f['file_description']))
    print('To download one of these files insert the file name '
          'as the first parameter to nussl.download_audio_file, like so: \n'
          ' >>> nussl.download_audio_file(\'K0140.wav\')')


def get_available_trained_models():
    """

    Returns:

    """
    return _download_metadata_file(constants.NUSSL_EFZ_MODEL_METADATA_URL)


def print_available_trained_models():
    """gets a list of available audio files for download from the server and displays them
    to the user.

    Args:

    Returns:

    Example:
        >>> import nussl
        >>> nussl.efz_utils.print_available_trained_models()
        File Name                                For Class            Size       Description
        deep_clustering_model.model              DeepClustering       48.1MiB    example Deep Clustering model
        deep_clustering_vocal_44k_long.model     DeepClustering       90.2MiB    trained DC model for vocal extraction

        Last updated 2018-03-06

    """
    file_metadata = get_available_trained_models()

    print('{:40} {:20} {:10} {:50}'.format('File Name', 'For Class', 'Size', 'Description'))
    for f in file_metadata:
        print('{:40} {:20} {:10} {:50}'.format(f['file_name'], f['for_class'],
                                               f['file_size'], f['file_description']))
    print('To download one of these files insert the file name '
          'as the first parameter to nussl.download_trained_model, like so: \n'
          ' >>> nussl.download_trained_model(\'deep_clustering_model.h5\')')


def get_available_benchmark_files():
    """

    Returns:

    """
    return _download_metadata_file(constants.NUSSL_EFZ_BENCHMARK_METADATA_URL)


def print_available_benchmark_files():
    """gets a list of available audio files for download from the server and displays them
    to the user.

    Args:

    Returns:

    Example:
        nussl.utils.print_available_benchmark_files()

        File Name                                For Class            Size       Description
        benchmark.npy                            example              11.0B      example benchmark file
        example.npy                              test                 13.0B      test example

        Last updated 2018-03-06

    """
    file_metadata = get_available_benchmark_files()

    print('{:40} {:20} {:10} {:50}'.format('File Name', 'For Class', 'Size', 'Description'))
    for f in file_metadata:
        print('{:40} {:20} {:10} {:50}'.format(f['file_name'], f['for_class'],
                                               f['file_size'], f['file_description']))
    print('To download one of these files insert the file name '
          'as the first parameter to nussl.download_benchmark_file, like so: \n'
          ' >>> nussl.download_benchmark_file(\'example.npy\')')


def _download_metadata_file(url):
    request = Request(url)

    # Make sure to get the newest data
    request.add_header('Pragma', 'no-cache')
    request.add_header('Cache-Control', 'max-age=0')
    response = urlopen(request)
    return json.loads(response.read())


def _download_metadata(file_name, file_type):
    """Downloads the metadata file for the specified file type and finds the entry for
    the specified file.

    Args:
        file_name: (String) name of the file's metadata to locate
        file_type: (String) type of file, determines the JSON file to download.
        One of [audio, model, benchmark].

    Returns:
        (dict) metadata for the specified file, or None if it could not be located.


    """

    metadata_urls = {
        'audio': constants.NUSSL_EFZ_AUDIO_METADATA_URL,
        'benchmark': constants.NUSSL_EFZ_BENCHMARK_METADATA_URL,
        'model': constants.NUSSL_EFZ_MODEL_METADATA_URL,
    }

    if metadata_urls[file_type]:
        metadata_url = metadata_urls[file_type]
    else:
        # wrong file type, return
        raise MetadataError('Cannot find metadata of type {}.'.format(file_type))

    metadata = _download_metadata_file(metadata_url)

    for file_metadata in metadata:
        if file_metadata['file_name'] == file_name:
            return file_metadata

    raise MetadataError('No matching metadata for file {} at url {}!'
                        .format(file_name, constants.NUSSL_EFZ_AUDIO_METADATA_URL))


def download_audio_file(example_name, local_folder=None, verbose=True):
    """Downloads the an audio file from the `nussl` server.

    Args:
        example_name: (String) name of the audio file to download
        local_folder: (String) path to local folder in which to download the file,
            defaults to regular `nussl` directory
        verbose: (bool)

    Returns:
        (String) path to the downloaded file

    Example:
        input_file = nussl.utils.download_audio_file('K0140.wav')
        music = AudioSignal(input_file, offset=45, duration=20)

    """
    file_metadata = _download_metadata(example_name, 'audio')

    file_hash = file_metadata['file_hash']

    file_url = urljoin(constants.NUSSL_EFZ_AUDIO_URL, example_name)
    result = _download_file(example_name, file_url, local_folder, 'audio',
                            file_hash=file_hash, verbose=verbose)

    return result


def download_trained_model(model_name, local_folder=None, verbose=True):
    """Downloads the a pre-trained model file from the `nussl` server.

    Args:
        model_name (str): name of the trained model to download
        local_folder (str):  a local folder in which to download the file,
        defaults to regular nussl directory
        verbose (bool):

    Returns:
        (String) path to the downloaded file

    Example:
        model_path = nussl.utils.download_trained_model('deep_clustering_vocal_44k_long.model')

    """
    file_metadata = _download_metadata(model_name, 'model')

    file_hash = file_metadata['file_hash']

    file_url = urljoin(constants.NUSSL_EFZ_MODELS_URL, model_name)
    result = _download_file(model_name, file_url, local_folder, 'models',
                            file_hash=file_hash, verbose=verbose)

    return result


def download_benchmark_file(benchmark_name, local_folder=None, verbose=True):
    """Downloads the specified Benchmark file from the `nussl` server.

    Args:
        benchmark_name (string): Name of the benchmark file to download.
        local_folder (string):  The local folder where the file will be downloaded.
        defaults to regular nussl directory
        verbose (bool): If True, will print updates as the file downloads

    Returns:
        (string) Path to the downloaded file.

    """
    file_metadata = _download_metadata(benchmark_name, 'benchmark')

    file_hash = file_metadata['file_hash']

    file_url = urljoin(constants.NUSSL_EFZ_BENCHMARKS_URL, benchmark_name)
    result = _download_file(benchmark_name, file_url, local_folder, 'benchmarks',
                            file_hash=file_hash, verbose=verbose)

    return result


def _download_file(file_name, url, local_folder, cache_subdir,
                   file_hash=None, cache_dir=None, verbose=True):
    """

    Heavily inspired by and lovingly adapted from keras' `get_file` function:
    https://github.com/fchollet/keras/blob/afbd5d34a3bdbb0916d558f96af197af1e92ce70/keras/utils/data_utils.py#L109

    Args:
        file_name: (String) name of the file located on the server
        url: (String) url of the file
        local_folder: (String) alternate folder in which to download the file
        cache_subdir: (String) subdirectory of folder in which to download flie
        file_hash: (String) expected hash of downloaded file
        cache_dir:

    Returns:
        (String) local path to downloaded file

    """
    if local_folder not in [None, '']:
        # local folder provided, let's create it if it doesn't exist and use it as datadir
        if not os.path.exists(os.path.expanduser(local_folder)):
            os.makedirs(os.path.expanduser(local_folder))
        datadir = os.path.expanduser(local_folder)

    else:
        if cache_dir is None:
            cache_dir = os.path.expanduser(os.path.join('~', '.nussl'))
        datadir_base = os.path.expanduser(cache_dir)

        if not os.access(datadir_base, os.W_OK):
            datadir_base = os.path.join('/tmp', '.nussl')

        datadir = os.path.join(datadir_base, cache_subdir)
        if not os.path.exists(datadir):
            os.makedirs(datadir)

    file_path = os.path.join(datadir, file_name)

    download = False
    if os.path.exists(file_path):
        if file_hash is not None:
            # compare the provided hash with the hash of the file currently at file_path
            current_hash = _hash_file(file_path)

            # if the hashes are equal, we already have the file we need, so don't download
            if file_hash != current_hash:
                if verbose:
                    warnings.warn("Hash for {} does not match known hash. "
                                  "Downloading {} from servers...".format(file_path, file_name))
                download = True
            elif verbose:
                print('Matching file found at {}, skipping download.'.format(file_path))

        else:
            download = True

    else:
        download = True

    if download:
        if verbose:
            print('Saving file at {}'.format(file_path))
            print('Downloading {} from {}'.format(file_name, url))

        def _dl_progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)

            if percent <= 100:
                sys.stdout.write('\r{}...{}%'.format(file_name, percent))
                sys.stdout.flush()

        error_msg = 'URL fetch failure on {}: {} -- {}'

        try:
            try:
                reporthook = _dl_progress if verbose else None
                urlretrieve(url, file_path, reporthook)
                if verbose: print()  # print a new line after the progress is done.

            except HTTPError as e:
                raise FailedDownloadError(error_msg.format(url, e.code, e.msg))
            except URLError as e:
                raise FailedDownloadError(error_msg.format(url, e.errno, e.reason))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e

        # check hash of received file to see if it matches the provided hash
        if file_hash is not None:
            download_hash = _hash_file(file_path)
            if file_hash != download_hash:
                # the downloaded file is not what it should be. Get rid of it.
                os.remove(file_path)
                raise MismatchedHashError("Downloaded file ({}) has been deleted "
                                          "because of a hash mismatch.".format(file_path))

        return file_path

    else:
        return file_path


def _hash_file(file_path, chunk_size=65535):
    """

    Args:
        file_path: System path to the file to be hashed
        chunk_size: size of chunks

    Returns:
        file_hash: the SHA256 hashed string in hex

    """
    hasher = hashlib.sha256()

    with open(file_path, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


# This is stolen wholesale from keras
if sys.version_info[0] == 2:
    def urlretrieve(url, filename, reporthook=None, data=None):
        """Replacement for `urlretrive` for Python 2.
        Under Python 2, `urlretrieve` relies on `FancyURLopener` from legacy
        `urllib` module, known to have issues with proxy management.
        # Arguments
            url: url to retrieve.
            filename: where to store the retrieved data locally.
            reporthook: a hook function that will be called once
                on establishment of the network connection and once
                after each block read thereafter.
                The hook will be passed three arguments;
                a count of blocks transferred so far,
                a block size in bytes, and the total size of the file.
            data: `data` argument passed to `urlopen`.
        """
        def chunk_read(response, chunk_size=8192, reporthook=None):
            content_type = response.info().get('Content-Length')
            total_size = -1
            if content_type is not None:
                total_size = int(content_type.strip())
            count = 0
            while 1:
                chunk = response.read(chunk_size)
                count += 1
                if not chunk:
                    reporthook(count, total_size, total_size)
                    break
                if reporthook:
                    reporthook(count, chunk_size, total_size)
                yield chunk

        response = urlopen(url, data)
        with open(filename, 'wb') as fd:
            for chunk in chunk_read(response, reporthook=reporthook):
                fd.write(chunk)
else:
    from six.moves.urllib.request import urlretrieve  # pylint: disable=g-import-not-at-top


########################################
#             Error Classes
########################################


class FailedDownloadError(Exception):
    """
    Exception class for failed file downloads.
    """
    pass


class MismatchedHashError(Exception):
    """
    Exception class for when a computed hash function does match a pre-computed hash.
    """
    pass


class MetadataError(Exception):
    """
    Exception class for errors with metadata.
    """
    pass
