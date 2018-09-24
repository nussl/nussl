#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
While *nussl* does not come with any data sets, it does have the capability to interface with
many common source separation data sets used within the MIR and speech separation communities.
These data set "hooks" are implemented as generator functions, and allow the user to loop through
each file in the data set and get separation :class:`AudioSignal` objects for the mixture and
individual sources.
"""
import os
import hashlib
import warnings

import numpy as np

import efz_utils
from audio_signal import AudioSignal

__all__ = ['iKala', 'mir1k', 'timit', 'medleyDB', 'musdb18', 'dsd100']


def _hash_directory(directory, ext=None):
    """
    Calculates the hash of every child file in the given directory using python's built-in SHA256
    function (using `os.walk()`, which also searches subdirectories recursively). If :param:`ext`
    is specified, this will only look at files with extension provided.

    This function is used to verify the integrity of data sets for use with nussl. Pretty much
    just makes sure that when we loop through/look at a directory, we understand the structure
    because the organization of the data set directories for different data sets are all unique
    and thus need to be hard coded by each generator function (below). If we get a hash mismatch
    we can throw an error easily.

    Args:
        directory (str): Directory within which file hashes get calculated. Searches recursively.
        ext (str): If provided, this function will only calculate the hash on files with the given
            extension.

    Returns:
        (str): String containing only hexadecimal digits of the has of the
            contents of the given directory.

    """

    hash_list = []
    for path, sub_dirs, files in os.walk(directory):
        if ext is None:
            hash_list.extend([efz_utils._hash_file(os.path.join(path, f)) for f in files
                              if os.path.isfile(os.path.join(path, f))])
        else:
            hash_list.extend([efz_utils._hash_file(os.path.join(path, f)) for f in files
                              if os.path.isfile(os.path.join(path, f))
                              if os.path.splitext(f)[1] == ext])

    hasher = hashlib.sha256()
    for hash_val in sorted(hash_list):  # Sort this list so we're platform agnostic
        hasher.update(hash_val.encode('utf-8'))

    return hasher.hexdigest()


def _check_hash(audio_dir, check_hash, expected_hash, ext):
    """
    Checks to see if the hashed contents of :param:`audio_dir` match :param:`expected_hash`. If
    :param:`check_hash` is ``False``, this function does nothing. If :param:`check_hash` is
    ``'warn'`` (a string) and there is a hash mismatch, then nussl will print a warning
    message to the console. If :param:`check_hash` is ``True`` and there is a mismatch, this
    function will throw a :class:`DataSetException`.

    Args:
        audio_dir (str): Top level directory to check the hash.
        check_hash (bool, str): In the case that there is a mismatch between the expected and
            calculated hash, if this parameter is ``True`` (a bool) an exception is raised and
            if this parameter is ``'warn'`` (a string) a warning is printed to the console. If
            this parameter is ``False``, the hash will not be calculated for this directory, i.e.,
            this function does nothing.
        expected_hash (str): The known, pre-computed hash that this function will test against.
        ext (str): When calculating the hash, only look at files with the provided extension.

    """
    # Check to see if the contents are what we expect
    # by hashing all of the files and checking against a precomputed expected_hash
    if check_hash and _hash_directory(audio_dir, ext) != expected_hash:
        msg = 'Hash of {} does not match known directory hash!'.format(audio_dir)
        if check_hash == 'warn':
            warnings.warn(msg)
        else:
            raise DataSetException(msg)


def _data_set_setup(directory, top_dir_name, audio_dir_name, expected_hash, check_hash, ext):
    """
    This function does some preliminary checking to make sure the directory we get from the user
    is oriented how we expect, checks the hash of the directory (if :param:`check_hash` is ``True``
    or ``'warn'``), and then spits out a list of the full path names of every audio file.

    Args:
        directory (str): The directory containing of the directory of the data set.
        top_dir_name (str): Name of the top-level directory of the data set. This should be the one
            that is downloaded from the data set host.
        audio_dir_name (str): Name of the directory that contains the audio files.
        expected_hash (str): Pre-calculated hash that this function will test against.
        check_hash (bool, str): In the case that there is a mismatch between the expected and
            calculated hash, if this parameter is ``True`` (a bool) an exception is raised and
            if this parameter is ``'warn'`` (a string) a warning is printed to the console. If
            this parameter is ``False``, the hash will not be calculated for this directory, i.e.,
            this function does nothing.
        ext (str): When calculating the hash, only look at files with the provided extension.

    Returns:
        (list): List of full paths for every audio file in the data set directory.
    """

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

    _check_hash(audio_dir, check_hash, expected_hash, ext)

    # return a list of the full paths of every audio file
    return [os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
            if os.path.isfile(os.path.join(audio_dir, f))
            if os.path.splitext(f)[1] == ext]


def _subset_and_shuffle(file_list, subset, shuffle, seed):
    """
    Given a list of file paths, this function will shuffle and make subsets of the given list.

    Args:
        file_list (list): List of full paths of audio files.
        subset (float, list, str, None): This parameter determines how to make a subset of the
        audio files in the data set. There are four ways to use it, depending on what type
        this parameter takes:
        1) If :param:`subset` is a ``float``, then :param:`subset` will return the first
           ``X.Y%`` of audio files, where ``X.Y%`` is some arbitrary percentage. In this case,
           :param:`subset` is expected to be in the range [0.0, 1.0].
        2) If :param:`subset` is a ``list``, it is expected to be a list of indices (as
           ``int``s). This function will then produce the audio files in the list that correspond
           to those indices.
        3) If :param:`subset` is a ``str``, it will only include audio files with that string
          somewhere in the directory name.
        4) If :param:`subset` is ``None``, then the whole data set is traversed unobstructed.

        shuffle (bool): Whether the data set should be shuffled.
        seed (int, 1-d array_like): Seed for ``numpy``'s random number generator used for
            shuffling.

    Returns:
        (list): Resultant list of file paths that is either shuffled and/or subset according to the
        input parameters.

    """

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

    elif isinstance(subset, (str, bytes)):
        result = [f for f in file_list if subset in os.path.dirname(f)]

    elif subset is None:
        result = file_list

    else:
        raise DataSetException('subset must be a list of indices or float between [0.0, 1.0]!')

    return result


def iKala(directory, check_hash=True, subset=None, shuffle=False, seed=None):
    """
    Generator function for the iKala data set. This allows you to loop through the entire data set
    with only a few :class:`AudioSignal` objects stored in memory at a time. There are options for
    only looping through a subset of the data set and shuffling the data set (with a seed). See
    details about those options below.

    `nussl` calculates the hash of the iKala directory and compares it against a precomputed hash
    for iKala that ships with `nussl`. This hash is used to verify that `nussl` understands the
    directory structure when reading the files. Calculating the hash can be turned off if the
    user needs a speed up, but this might cause oblique errors if the iKala directory is not set up
    in the same way as a fresh download of iKala.

    Examples:
        Using this generator function to loop through the iKala data set. In this example, we use
        the generator directly in the ``for`` loop.

        .. code-block:: python
            :linenos:

            iKala_path = '/path/to/iKala'  # the iKala directory in disc
            for mix, vox, acc in nussl.datasets.iKala(iKala_path):
                mix.to_mono(overwrite=True)  # sum to mono to make a 'mixture'

                # Get some basic metadata on the files.
                # (They'll all have the same file name, but different labels)
                print('Mixture       - Filename: {}, Label: {}'.format(mix.file_name, mix.label))
                print('Vocals        - Filename: {}, Label: {}'.format(vox.file_name, vox.label))
                print('Accompaniment - Filename: {}, Label: {}'.format(acc.file_name, acc.label))

                # Run an algorithm on the iKala files and save to disc
                r = nussl.Repet(mix)
                r.run()
                bg_est, fg_est = r.make_audio_signals()
                bg_est.write_audio_to_file('{}_bg.wav'.format(os.path.splitext(mix.file_name)[0]))
                fg_est.write_audio_to_file('{}_fg.wav'.format(os.path.splitext(mix.file_name)[0]))

        It's also possible to use ``tqdm`` to print the progress to the console. This is useful
        because running through an entire data set can take a while. Here's a more advanced example
        using some other options as well:

        .. code-block:: python
            :linenos:

            import nussl
            import tdqm

            iKala_path = 'path/to/iKala' # the iKala directory on disc
            idxs = range(29, 150)[::2]  # Only get every other song between [29, 150)
            iKala_gen = nussl.datasets.iKala(iKala_path, subset=idxs, check_hash=False)

            # Tell tqdm the number of files we're running on so it can estimate a completion time
            for mixture, vocals, accompaniment in tqdm(iKala_gen, total=len(idxs)):
                mix.to_mono(overwrite=True)  # sum to mono to make a 'mixture'

                # Run an algorithm on the iKala files and save to disc
                r = nussl.Repet(mix)
                r.run()
                bg_est, fg_est = r.make_audio_signals()
                bg_est.write_audio_to_file('{}_bg.wav'.format(os.path.splitext(mix.file_name)[0]))
                fg_est.write_audio_to_file('{}_fg.wav'.format(os.path.splitext(mix.file_name)[0]))

    Args:
        directory (str): Top-level directory for the iKala data set.
        check_hash (bool, str): In the case that there is a mismatch between the expected and
        calculated hash, if this parameter is ``True`` (a bool) an exception is raised and
        if this parameter is ``'warn'`` (a string) a warning is printed to the console. If
        this parameter is ``False``, the hash will not be calculated for this directory, i.e.,
        this function does nothing.
        subset (float, list, str, None): This parameter determines how to make a subset of the
        audio files in the data set. There are four ways to use it, depending on what type
        this parameter takes:
        1) If :param:`subset` is a ``float``, then :param:`subset` will return the first
           ``X.Y%`` of audio files, where ``X.Y%`` is some arbitrary percentage. In this case,
           :param:`subset` is expected to be in the range [0.0, 1.0].
        2) If :param:`subset` is a ``list``, it is expected to be a list of indices (as
           ``int``s). This function will then produce the audio files in the list that correspond
           to those indices.
        3) If :param:`subset` is a ``str``, it will only include audio files with that string
         somewhere in the directory name.
        4) If :param:`subset` is ``None``, then the whole data set is traversed unobstructed.
        shuffle (bool): Whether the data set should be shuffled.
        seed (int, 1-d array_like): Seed for ``numpy``'s random number generator used for
        shuffling.

    Yields:
        (``tuple(AudioSignal, AudioSignal, AudioSignal)``):
            A tuple of three :class:`AudioSignal` objects, with audio loaded for each source. In
            the tuple, they are returned in the following order:
            ``(mixture, vocals, accompaniment)``. In iKala, the audio files are such that the
            vocals are hard panned to one channel and the accompaniment is hard panned to the other.
            So, the 'mixture' yielded here by this function reflects this, and needs to 'mixed'
            down to mono. In other words, ``mixture`` is a stereo :class:`AudioSignal` object,
            where each channel is on source, and similarly ``vocals`` and ``accompaniment`` are
            mono :class:`AudioSignal` objects made from a single channel in `mixture`.

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
        mixture.label = 'mixture'

        vocals = mixture.make_audio_signal_from_channel(1)
        vocals.label = 'vocals'

        accompaniment = mixture.make_audio_signal_from_channel(0)
        accompaniment.label = 'accompaniment'

        yield mixture, vocals, accompaniment


def mir1k(directory, check_hash=True, subset=None, shuffle=False, seed=None, undivided=False):
    """
    Generator function for the MIR-1K data set. This allows you to loop through the entire data set
    with only a few :class:`AudioSignal` objects stored in memory at a time. There are options for
    only looping through a subset of the data set and shuffling the data set (with a seed). See
    details about those options below.

    `nussl` calculates the hash of the MIR-1K directory and compares it against a precomputed hash
    for MIR-1K that ships with `nussl`. This hash is used to verify that `nussl` understands the
    directory structure when reading the files. Calculating the hash can be turned off if the
    user needs a speed up, but this might cause oblique errors if the MIR-1K directory is not set up
    in the same way as a fresh download of MIR-1K.

    MIR-1K also ships with two 'sets' of audio files: the divided and undivided sets. They contain
    the same content, the only difference is that the undivided set is one file per song, each song
    taking up the whole file, and the divided set has the same song divided into segments of ~3-12
    seconds. The :param:`undivided` parameter controls which of these two sets `nussl` will loop
    through.

    Examples:
        Using this generator function to loop through the MIR-1K data set. In this example, we use
        the generator directly in the ``for`` loop.

        .. code-block:: python
            :linenos:

            mir1k_path = '/path/to/MIR-1K'  # the MIR-1K directory in disc
            for mix, vox, acc in nussl.datasets.mir1k(mir1k_path):
                mix.to_mono(overwrite=True)  # sum to mono to make a 'mixture'

                # Get some basic metadata on the files.
                # (They'll all have the same file name, but different labels)
                print('Mixture       - Filename: {}, Label: {}'.format(mix.file_name, mix.label))
                print('Vocals        - Filename: {}, Label: {}'.format(vox.file_name, vox.label))
                print('Accompaniment - Filename: {}, Label: {}'.format(acc.file_name, acc.label))

                # Run an algorithm on the MIR-1K files and save to disc
                r = nussl.Repet(mix)
                r.run()
                bg_est, fg_est = r.make_audio_signals()
                bg_est.write_audio_to_file('{}_bg.wav'.format(os.path.splitext(mix.file_name)[0]))
                fg_est.write_audio_to_file('{}_fg.wav'.format(os.path.splitext(mix.file_name)[0]))

        It's also possible to use ``tqdm`` to print the progress to the console. This is useful
        because running through an entire data set can take a while. Here's a more advanced example
        using some other options as well:

        .. code-block:: python
            :linenos:

            import nussl
            import tdqm

            mir1k_path = 'path/to/MIR-1K' # the MIR-1K directory on disc
            idxs = range(29, 150)[::2]  # Only get every other song between [29, 150)
            mir1k_gen = nussl.datasets.mir1k(mir1k_path, subset=idxs,
                                             check_hash=False, undivided=True)

            # Tell tqdm the number of files we're running on so it can estimate a completion time
            for mixture, vocals, accompaniment in tqdm(mir1k_gen, total=len(idxs)):
                mix.to_mono(overwrite=True)  # sum to mono to make a 'mixture'

                # Run an algorithm on the MIR-1K files and save to disc
                r = nussl.Repet(mix)
                r.run()
                bg_est, fg_est = r.make_audio_signals()
                bg_est.write_audio_to_file('{}_bg.wav'.format(os.path.splitext(mix.file_name)[0]))
                fg_est.write_audio_to_file('{}_fg.wav'.format(os.path.splitext(mix.file_name)[0]))

    Args:
        directory (str): Top-level directory for the MIR-1K data set.
        check_hash (bool, str): In the case that there is a mismatch between the expected and
        calculated hash, if this parameter is ``True`` (a bool) an exception is raised and
        if this parameter is ``'warn'`` (a string) a warning is printed to the console. If
        this parameter is ``False``, the hash will not be calculated for this directory, i.e.,
        this function does nothing.
        subset (float, list, str, None): This parameter determines how to make a subset of the
        audio files in the data set. There are four ways to use it, depending on what type
        this parameter takes:
        1) If :param:`subset` is a ``float``, then :param:`subset` will return the first
        ``X.Y%`` of audio files, where ``X.Y%`` is some arbitrary percentage. In this case,
        :param:`subset` is expected to be in the range [0.0, 1.0].
        2) If :param:`subset` is a ``list``, it is expected to be a list of indices (as
        ``int``s). This function will then produce the audio files in the list that correspond
        to those indices.
        3) If :param:`subset` is a ``str``, it will only include audio files with that string
        somewhere in the directory name.
        4) If :param:`subset` is ``None``, then the whole data set is traversed unobstructed.
        shuffle (bool): Whether the data set should be shuffled.
        seed (int, 1-d array_like): Seed for ``numpy``'s random number generator used for
        shuffling.
        undivided (bool): Whether to use the divided (in the ``Wavefile`` directory) or undivided
        (in the ``UndividedWavefile`` directory).

    Yields:
        (``tuple(AudioSignal, AudioSignal, AudioSignal)``):
            A tuple of three :class:`AudioSignal` objects, with audio loaded for each source. In
            the tuple, they are returned in the following order:
            ``(mixture, vocals, accompaniment)``. In MIR-1K, the audio files are such that the
            vocals are hard panned to one channel and the accompaniment is hard panned to the other.
            So, the 'mixture' yielded here by this function reflects this, and needs to 'mixed'
            down to mono. In other words, ``mixture`` is a stereo :class:`AudioSignal` object,
            where each channel is on source, and similarly ``vocals`` and ``accompaniment`` are
            mono :class:`AudioSignal` objects made from a single channel in `mixture`.

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
        mixture.label = 'mixture'

        vocals = mixture.make_audio_signal_from_channel(1)
        vocals.label = 'vocals'

        accompaniment = mixture.make_audio_signal_from_channel(0)
        accompaniment.label = 'accompaniment'

        yield mixture, vocals, accompaniment


def timit(directory, check_hash=True, subset=None, shuffle=False, seed=None):
    """
    Not implemented yet.

    Args:
        directory:
        check_hash:
        subset:
        shuffle:
        seed:

    Yields:

    """
    dir_name = 'TIMIT'
    hash = ''
    audio_extension = '.wav'
    all_wav_files = _data_set_setup(directory, dir_name, dir_name,
                                    hash, check_hash, audio_extension)

    all_wav_files = _subset_and_shuffle(all_wav_files, subset, shuffle, seed)


def medleyDB(directory, raw=False, check_hash=True, subset=None, shuffle=False, seed=None):
    """
    Not implemented yet.

    Args:
        directory:
        check_hash:
        subset:
        shuffle:
        seed:

    Returns:

    """
    top_dir_name = 'medleyDB'
    audio_dir_name = 'Audio'
    hash = ''
    audio_extension = '.wav'


def dsd100(directory, check_hash=True, subset=None, shuffle=False, seed=None):
    """
    Not implemented yet.

    Args:
        directory:
        check_hash:
        subset:
        shuffle:
        seed:

    Returns:

    """


def musdb18(directory, check_hash=True, subset=None, folder=None, shuffle=False, seed=None):
    """
    Generator function for the MUSDB18 data set. This allows you to loop through the entire data set
    with only a few :class:`AudioSignal` objects stored in memory at a time. There are options for
    only looping through a subset of the data set and shuffling the data set (with a seed). See
    details about those options below.

    `nussl` calculates the hash of the MUSDB18 directory and compares it against a precomputed hash
    for MUSDB18 that ships with `nussl`. This hash is used to verify that `nussl` understands the
    directory structure when reading the files. Calculating the hash can be turned off if the
    user needs a speed up, but this might cause oblique errors if the MUSDB directory is not set up
    in the same way as a fresh download of MUSDB18.

    The audio in MUSDB18 is stored in the 'stempeg' format from Native Instruments. `nussl` uses
    the `stempeg` library to read these files from disc, and returns each of the sources as
    individual :obj:`AudioSignal` objects.

    Examples:
        Using this generator function to loop through the MUSDB18 data set. In this example, we use
        the generator directly in the ``for`` loop.

        .. code-block:: python
            :linenos:

            musdb_path = '/path/to/MUSDB18'  # the MUSDB18 directory in disc
            for mix, drums, bass, other, vox in nussl.datasets.musdb(musdb_path):

                # Get some basic metadata on the files.
                # (They'll all have the same file name, but different labels)
                print('Mixture  - Filename: {}, Label: {}'.format(mix.file_name, mix.label))
                print('Vocals   - Filename: {}, Label: {}'.format(vox.file_name, vox.label))
                print('Drums    - Filename: {}, Label: {}'.format(drums.file_name, drums.label))

                # Run an algorithm on the MUSDB18 files and save to disc
                r = nussl.Repet(mix)
                r.run()
                bg_est, fg_est = r.make_audio_signals()
                bg_est.write_audio_to_file('{}_bg.wav'.format(os.path.splitext(mix.file_name)[0]))
                fg_est.write_audio_to_file('{}_fg.wav'.format(os.path.splitext(mix.file_name)[0]))

        It's also possible to use ``tqdm`` to print the progress to the console. This is useful
        because running through an entire data set can take a while. Here's a more advanced example
        using some other options as well:

        .. code-block:: python
            :linenos:

            import nussl
            import tdqm

            musdb_path = 'path/to/MUSDB18' # the MUSDB18 directory on disc

            # Only run on the 'test' folder (this has 50 songs)
            musdb_gen = nussl.datasets.musdb(musdb_path, subset='test', check_hash=False)

            # Tell tqdm the number of files we're running on so it can estimate a completion time
            for mix, drums, bass, other, vox in tqdm(musdb_gen, total=50):

                # Run an algorithm on the MUSDB18 files and save to disc
                r = nussl.Repet(mix)
                r.run()
                bg_est, fg_est = r.make_audio_signals()
                bg_est.write_audio_to_file('{}_bg.wav'.format(os.path.splitext(mix.file_name)[0]))
                fg_est.write_audio_to_file('{}_fg.wav'.format(os.path.splitext(mix.file_name)[0]))

    Args:
        directory (str): Top-level directory for the MUSDB18 data set.
        check_hash (bool, str): In the case that there is a mismatch between the expected and
            calculated hash, if this parameter is ``True`` (a bool) an exception is raised and
            if this parameter is ``'warn'`` (a string) a warning is printed to the console. If
            this parameter is ``False``, the hash will not be calculated for this directory, i.e.,
            this function does nothing.
        subset (float, list, str, None): This parameter determines how to make a subset of the
        audio files in the data set. There are four ways to use it, depending on what type
        this parameter takes:
        * If :param:`subset` is a ``float``, then :param:`subset` will return the first
        ``X.Y%`` of audio files, where ``X.Y%`` is some arbitrary percentage. In this case,
        :param:`subset` is expected to be in the range [0.0, 1.0].
        ( If :param:`subset` is a ``list``, it is expected to be a list of indices (as
        ``int``s). This function will then produce the audio files in the list that correspond
        to those indices.
        ( If :param:`subset` is a ``str``, it will only include audio files with that string
        somewhere in the directory name.
        * If :param:`subset` is ``None``, then the whole data set is traversed unobstructed.
        shuffle (bool): Whether the data set should be shuffled.
        seed (int, 1-d array_like): Seed for ``numpy``'s random number generator used for
        shuffling.

    Yields:
        (tuple):
            A tuple of five :class:`AudioSignal` objects, with audio loaded for each source. In
            the tuple, they are returned in the following order:
            ``(mixture, drums, bass, other, vox)``.
    """

    try:
        import stempeg
    except Exception:
        raise ImportError('Cannot read MUSDB18 without stempeg package installed!')

    dir_name = 'musdb18'
    audio_dir_names = ['test', 'train'] if folder not in ('test', 'train') else [folder]
    hash = 'cf4cfcef4eadc212c34df6e8fb1184a3f63b7fedfab23e79d17e735fff0bfaf9'
    audio_extension = '.mp4'
    stem_labels = ['mixture', 'drums', 'bass', 'other', 'vocals']

    # Check the hash has of the full directory
    _check_hash(directory, check_hash, hash, audio_extension)

    files = []
    for audio_dir_name in audio_dir_names:
        files.extend(_data_set_setup(directory, dir_name, audio_dir_name,
                                     '', False, audio_extension))

    files = _subset_and_shuffle(files, subset, shuffle, seed)

    for f in files:
        stem, sr = stempeg.read_stems(f)
        yield tuple(_make_audio_signal_from_stem(f, stem, i, sr, l)
                    for i, l in enumerate(stem_labels))


def _make_audio_signal_from_stem(filename, stem, i, sr, label):
    """
    Reads in a :param:`stem` from the stempeg library (`stempeg.read_stems()`) and creates a
    correctly formatted :class:`AudioSignal` object (with all the metadata set up).
    Args:
        filename (str): Name of the file on disc.
        stem (:obj:`np.ndarray`): Numpy array from the `stempeg.read_stems()` function.
        i (int): Index of the :param:`stem: array to get audio data from.
        sr (int): Sample rate.
        label (str): Label for the :class:`AudioSignal` object.

    Returns:
        (:obj:`AudioSignal`) Correctly formatted :class:`AudioSignal` object with the
            right metadata.

    """
    signal = AudioSignal(audio_data_array=stem[i,...], sample_rate=sr)
    signal.path_to_input_file = filename
    signal.label = label
    return signal


class DataSetException(Exception):
    """
    Exception class for errors when working with data sets in nussl.
    """
    pass
