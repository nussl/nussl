#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The *nussl* External File Zoo (EFZ) is a server that houses all files that are too large to
bundle with *nussl* when distributing it through ``pip`` or Github. These types of files include
audio examples, benchmark files for tests, and trained neural network models.

*nussl* has built-in utilities for accessing the EFZ through its API. Here, it is possible to
see what files are available on the EFZ and download desired files. The EFZ utilities allow
for such functionality.
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
    Returns a list of dicts containing metadata of the available audio files on the nussl External
    File Zoo (EFZ) server (http://nussl.ci.northwestern.edu/).

    Each entry in the list is in the following format:

    .. code-block:: python

        {
            u'file_length_seconds': 5.00390022675737,
            u'visible': True,
            u'file_name': u'K0140.wav',
            u'date_modified': u'2018-06-01',
            u'file_hash': u'f0d8d3c8d199d3790b0e42d1e5df50a6801f928d10f533149ed0babe61b5d7b5',
            u'file_size_bytes': 441388,
            u'file_description': u'Acoustic piano playing middle C.',
            u'audio_attributes': u'piano, middle C',
            u'file_size': u'431.0KiB',
            u'date_added': u'2018-06-01'
        }

    See Also:
        * :func:`print_available_audio_files`, prints a list of the audio files to the console.
        * :func:`download_audio_file` to download an audio file from the EFZ.

    Returns:
        (list): A list of dicts containing metadata of the available audio files on the nussl
        External File Zoo (EFZ) server (http://nussl.ci.northwestern.edu/).

    """
    # _download_all_metadata() will throw its own errors, so no try block around it
    return _download_all_metadata(constants.NUSSL_EFZ_AUDIO_METADATA_URL)


def print_available_audio_files():
    """
    Prints a message to the console that shows all of the available audio files that are on the
    nussl External File Zoo (EFZ) server (http://nussl.ci.northwestern.edu/).

    See Also:
        * :func:`get_available_audio_files` to get this same data from the EFZ server as a list.
        * :func:`download_audio_file` to download an audio file from the EFZ.

    Example:
        >>> import nussl
        >>> nussl.efz_utils.print_available_audio_files()
        File Name                                Duration (sec)  Size       Description
        dev1_female3_inst_mix.wav                10.0            1.7MiB     Instantaneous mixture of three female speakers talking in a stereo field.
        dev1_female3_synthconv_130ms_5cm_mix.wav 10.0            1.7MiB     Three female speakers talking in a stereo field, with 130ms of inter-channel delay.
        K0140.wav                                5.0             431.0KiB   Acoustic piano playing middle C.
        K0149.wav                                5.0             430.0KiB   Acoustic piano playing the A above middle C. (A440)

    To download one of these files insert the file name as the first parameter to
    :func:`download_audio_file`, like so:

    >>> nussl.efz_utils.download_audio_file('K0140.wav')

    """
    file_metadata = get_available_audio_files()

    print('{:40} {:15} {:10} {:50}'.format('File Name', 'Duration (sec)',
                                           'Size', 'Description'))
    for f in file_metadata:
        print('{:40} {:<15.1f} {:10} {:50}'.format(f['file_name'], f['file_length_seconds'],
                                                   f['file_size'], f['file_description']))
    print('To download one of these files insert the file name '
          'as the first parameter to nussl.download_audio_file(), like so: \n'
          ' >>> nussl.efz_utils.download_audio_file(\'K0140.wav\')')


def get_available_trained_models():
    """
    Returns a list of dicts containing metadata of the available trained models on the nussl
    External File Zoo (EFZ) server (http://nussl.ci.northwestern.edu/).

    Each entry in the list is in the following format:

    .. code-block:: python

        {
            u'for_class': u'DeepClustering',
            u'visible': True,
            u'file_name': u'deep_clustering_vocals_44k_long.model',
            u'date_modified': u'2018-06-01',
            u'file_hash': u'e09034c2cb43a293ece0b121f113b8e4e1c5a247331c71f40cb9ca38227ccc2c',
            u'file_size_bytes': 94543355,
            u'file_description': u'Deep clustering for vocal separation trained on augmented DSD100.',
            u'file_size': u'90.2MiB',
            u'date_added': u'2018-06-01'
        }

    Notes:
        Most of the entries in the dictionary are self-explanatory, but note the ``for_class``
        entry. The ``for_class`` entry specifies which `nussl` separation class the given model will
        work with. Usually, `nussl` separation classes that require a model will default so
        retrieving a model on the EFZ server (if not already found on the user's machine), but
        sometimes it is desirable to use a model other than the default one provided. In this case,
        the ``for_class`` entry lets the user know which class it is valid for use with.
        Additionally, trying to load a model into a class that it is not explicitly labeled for that
        class will raise an exception. Just don't do it, ok?

    See Also:
        * :func:`print_available_trained_models`, prints a list of the trained models to
            the console.
        * :func:`download_trained_model` to download a trained model from the EFZ.

    Returns:
        (list): A list of dicts containing metadata of the available trained models on the nussl
        External File Zoo (EFZ) server (http://nussl.ci.northwestern.edu/).
    """
    return _download_all_metadata(constants.NUSSL_EFZ_MODEL_METADATA_URL)


def print_available_trained_models():
    """
    Prints a message to the console that shows all of the available trained models that are on the
    nussl External File Zoo (EFZ) server (http://nussl.ci.northwestern.edu/).

    Notes:
        Most of the entries in the dictionary are self-explanatory, but note the ``for_class``
        entry. The ``for_class`` entry specifies which `nussl` separation class the given model will
        work with. Usually, `nussl` separation classes that require a model will default so
        retrieving a model on the EFZ server (if not already found on the user's machine), but
        sometimes it is desirable to use a model other than the default one provided. In this case,
        the ``for_class`` entry lets the user know which class it is valid for use with.
        Additionally, trying to load a model into a class that it is not explicitly labeled for that
        class will raise an exception. Just don't do it, ok?

    See Also:
        * :func:`get_available_trained_models` to get this same data from the EFZ server as a list.
        * :func:`download_trained_model` to download a trained model from the EFZ.

    Example:
        >>> import nussl
        >>> nussl.efz_utils.print_available_trained_models()
        File Name                                For Class            Size       Description
        deep_clustering_model.model              DeepClustering       48.1MiB    example Deep Clustering model
        deep_clustering_vocal_44k_long.model     DeepClustering       90.2MiB    trained DC model for vocal extraction

    To download one of these files insert the file name as the first parameter to download_trained_model(), like so:

    >>> nussl.efz_utils.download_trained_model('deep_clustering_model.h5')

    """
    file_metadata = get_available_trained_models()

    print('{:40} {:20} {:10} {:50}'.format('File Name', 'For Class', 'Size', 'Description'))
    for f in file_metadata:
        print('{:40} {:20} {:10} {:50}'.format(f['file_name'], f['for_class'],
                                               f['file_size'], f['file_description']))
    print('To download one of these files insert the file name '
          'as the first parameter to nussl.download_trained_model, like so: \n'
          ' >>> nussl.efz_utils.download_trained_model(\'deep_clustering_model.h5\')')


def get_available_benchmark_files():
    """
    Returns a list of dicts containing metadata of the available benchmark files for tests on the
    nussl External File Zoo (EFZ) server (http://nussl.ci.northwestern.edu/).

    Each entry in the list is in the following format:

    .. code-block:: python

        {
            u'for_class': u'DuetUnitTests',
            u'visible': True, u'file_name':
            u'benchmark_atn_bins.npy',
            u'date_modified': u'2018-06-19',
            u'file_hash': u'cf7fef6f4ea9af3dbde8b9880602eeaf72507b6c78f04097c5e79d34404a8a1f',
            u'file_size_bytes': 488,
            u'file_description': u'Attenuation bins numpy array for DUET benchmark test.',
            u'file_size': u'488.0B',
            u'date_added': u'2018-06-19'
        }

    Notes:
        Most of the entries in the dictionary are self-explanatory, but note the `for_class`
        entry. The `for_class` entry specifies which `nussl` benchmark class will load the
        corresponding benchmark file. Make sure these match exactly when writing tests!

    See Also:
        * :func:`print_available_benchmark_files`, prints a list of the benchmark files to the
            console.
        * :func:`download_benchmark_file` to download an benchmark file from the EFZ.

    Returns:
        (list): A list of dicts containing metadata of the available audio files on the nussl
        External File Zoo (EFZ) server (http://nussl.ci.northwestern.edu/).

    """
    return _download_all_metadata(constants.NUSSL_EFZ_BENCHMARK_METADATA_URL)


def print_available_benchmark_files():
    """
    Prints a message to the console that shows all of the available benchmark files that are on the
    nussl External File Zoo (EFZ) server (http://nussl.ci.northwestern.edu/).

    Example:
        >>> import nussl
        >>> nussl.efz_utils.print_available_benchmark_files()
        File Name                                For Class            Size       Description
        mix3_matlab_repet_foreground.mat         TestRepet            6.4MiB     Foreground matrix for Repet class benchmark test.
        benchmark_atn_bins.npy                   DuetUnitTests        488.0B     Attenuation bins numpy array for DUET benchmark test.
        benchmark_sym_atn.npy                    DuetUnitTests        3.4MiB     Symmetric attenuation histogram for the DUET benchmark test.
        benchmark_wmat.npy                       DuetUnitTests        3.4MiB     Frequency matrix for the DUET benchmark test.

    To download one of these files insert the file name as the first parameter to nussl.download_benchmark_file, like so:

    >>> nussl.efz_utils.download_benchmark_file('example.npy')

    Notes:
        Most of the entries in the printed list are self-explanatory, but note the ``for_class``
        entry. The ``for_class`` entry specifies which `nussl` benchmark class will load the
        corresponding benchmark file. Make sure these match exactly when writing tests!

    See Also:
        * :func:`get_available_benchmark_files`, prints a list of the benchmark files to the
            console.
        * :func:`download_benchmark_file` to download an benchmark file from the EFZ.

    """
    file_metadata = get_available_benchmark_files()

    print('{:40} {:20} {:10} {:50}'.format('File Name', 'For Class', 'Size', 'Description'))
    for f in file_metadata:
        print('{:40} {:20} {:10} {:50}'.format(f['file_name'], f['for_class'],
                                               f['file_size'], f['file_description']))
    print('To download one of these files insert the file name '
          'as the first parameter to nussl.download_benchmark_file, like so: \n'
          ' >>> nussl.efz_utils.download_benchmark_file(\'example.npy\')')


def _download_all_metadata(url):
    """
    Downloads the json file that contains all of the metadata for a specific file type (read:
    audio files, benchmark files, or trained models) that is on the EFZ server. This is retrieved
    from one of following three URLs (which are stored in nussl.constants):
    NUSSL_EFZ_AUDIO_METADATA_URL, NUSSL_EFZ_BENCHMARK_METADATA_URL, or NUSSL_EFZ_MODEL_METADATA_URL.

    Args:
        url (str):  URL for the EFZ server that has metadata. One of these three:
            NUSSL_EFZ_AUDIO_METADATA_URL, NUSSL_EFZ_BENCHMARK_METADATA_URL, or
            NUSSL_EFZ_MODEL_METADATA_URL.

    Returns:
        (list): List of dicts with metadata for the desired file type.

    """
    request = Request(url)

    # Make sure to get the newest data
    request.add_header('Pragma', 'no-cache')
    request.add_header('Cache-Control', 'max-age=0')
    response = urlopen(request)
    return json.loads(response.read())


def _download_metadata_for_file(file_name, file_type):
    """
    Downloads the metadata entry for a specific file (:param:`file_name`) on the EFZ server.

    Args:
        file_name (str): File name as specified on the EFZ server.
        file_type (str): 'Type' of file, either 'audio', 'model', or 'benchmark'.

    Returns:
        (dict) Metadata entry for the specified file, or ``None`` if it could not be located.

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

    metadata = _download_all_metadata(metadata_url)

    for file_metadata in metadata:
        if file_metadata['file_name'] == file_name:
            return file_metadata

    raise MetadataError('No matching metadata for file {} at url {}!'
                        .format(file_name, constants.NUSSL_EFZ_AUDIO_METADATA_URL))


def download_audio_file(audio_file_name, local_folder=None, verbose=True):
    """
    Downloads the specified audio file from the `nussl` External File Zoo (EFZ) server. The
    downloaded file is stored in :param:`local_folder` if a folder is provided. If a folder is
    not provided, `nussl` attempts to save the downloaded file in `~/.nussl/` (expanded) or in
    `tmp/.nussl`. If the requested file is already in :param:`local_folder` (or one of the two
    aforementioned directories) and the calculated hash matches the precomputed hash from the EFZ
    server metadata, then the file will not be downloaded.

    Args:
        audio_file_name: (str) Name of the audio file to attempt to download.
        local_folder: (str) Path to local folder in which to download the file.
            If no folder is provided, `nussl` will store the file in `~/.nussl/` (expanded) or in
            `tmp/.nussl`.
        verbose (bool): If ``True`` prints the status of the download to the console.

    Returns:
        (String) Full path to the requested file (whether downloaded or not).

    Example:
        >>> import nussl
        >>> piano_path = nussl.efz_utils.download_audio_file('K0140.wav')
        >>> piano_signal = nussl.AudioSignal(piano_path)

    """
    file_metadata = _download_metadata_for_file(audio_file_name, 'audio')

    file_hash = file_metadata['file_hash']

    file_url = urljoin(constants.NUSSL_EFZ_AUDIO_URL, audio_file_name)
    result = _download_file(audio_file_name, file_url, local_folder, 'audio',
                            file_hash=file_hash, verbose=verbose)

    return result


def download_trained_model(model_name, local_folder=None, verbose=True):
    """
    Downloads the specified trained model from the `nussl` External File Zoo (EFZ) server. The
    downloaded file is stored in :param:`local_folder` if a folder is provided. If a folder is
    not provided, `nussl` attempts to save the downloaded file in `~/.nussl/` (expanded) or in
    `tmp/.nussl`. If the requested file is already in :param:`local_folder` (or one of the two
    aforementioned directories) and the calculated hash matches the precomputed hash from the EFZ
    server metadata, then the file will not be downloaded.

    Args:
        audio_file_name: (str) Name of the trained model to attempt to download.
        local_folder: (str) Path to local folder in which to download the file.
            If no folder is provided, `nussl` will store the file in `~/.nussl/` (expanded) or in
            `tmp/.nussl`.
        verbose (bool): If ``True`` prints the status of the download to the console.

    Returns:
        (String) Full path to the requested file (whether downloaded or not).

    Example:
        >>> import nussl
        >>> model_path = nussl.efz_utils.download_trained_model('deep_clustering_model.h5')
        >>> signal = nussl.AudioSignal()
        >>> piano_signal = nussl.DeepClustering(signal, model_path=model_path)

    """
    file_metadata = _download_metadata_for_file(model_name, 'model')

    file_hash = file_metadata['file_hash']

    file_url = urljoin(constants.NUSSL_EFZ_MODELS_URL, model_name)
    result = _download_file(model_name, file_url, local_folder, 'models',
                            file_hash=file_hash, verbose=verbose)

    return result


def download_benchmark_file(benchmark_name, local_folder=None, verbose=True):
    """
    Downloads the specified benchmark file from the `nussl` External File Zoo (EFZ) server. The
    downloaded file is stored in :param:`local_folder` if a folder is provided. If a folder is
    not provided, `nussl` attempts to save the downloaded file in `~/.nussl/` (expanded) or in
    `tmp/.nussl`. If the requested file is already in :param:`local_folder` (or one of the two
    aforementioned directories) and the calculated hash matches the precomputed hash from the EFZ
    server metadata, then the file will not be downloaded.

    Args:
        audio_file_name: (str) Name of the trained model to attempt to download.
        local_folder: (str) Path to local folder in which to download the file.
            If no folder is provided, `nussl` will store the file in `~/.nussl/` (expanded) or in
            `tmp/.nussl`.
        verbose (bool): If ``True`` prints the status of the download to the console.

    Returns:
        (String) Full path to the requested file (whether downloaded or not).

    Example:
        >>> import nussl
        >>> import numpy as np
        >>> stm_atn_path = nussl.efz_utils.download_benchmark_file('benchmark_sym_atn.npy')
        >>> sym_atm = np.load(stm_atn_path)

    """
    file_metadata = _download_metadata_for_file(benchmark_name, 'benchmark')

    file_hash = file_metadata['file_hash']

    file_url = urljoin(constants.NUSSL_EFZ_BENCHMARKS_URL, benchmark_name)
    result = _download_file(benchmark_name, file_url, local_folder, 'benchmarks',
                            file_hash=file_hash, verbose=verbose)

    return result


def _download_file(file_name, url, local_folder, cache_subdir,
                   file_hash=None, cache_dir=None, verbose=True):
    """
    Downloads the specified file from the

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
