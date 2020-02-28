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

from . import efz_utils
from .audio_signal import AudioSignal

__all__ = ['iKala', 'mir1k', 'timit', 'medleyDB', 'musdb18', 'dsd100']


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
        msg = f'Hash of {audio_dir} does not match known directory hash!'
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
        raise DataSetException(f"Expected directory, got '{directory}'!")

    if top_dir_name != os.path.split(directory)[1]:
        raise DataSetException(f"Expected {top_dir_name}, got '{os.path.split(directory)[1]}'")

    # This should be tha directory with all of the audio files
    audio_dir = os.path.join(directory, audio_dir_name)
    if not os.path.isdir(audio_dir):
        raise DataSetException(f'Expected {audio_dir} to be a directory but it is not!')

    _check_hash(audio_dir, check_hash, expected_hash, ext)

    # return a list of the full paths of every audio file
    return [os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
            if os.path.isfile(os.path.join(audio_dir, f))
            if os.path.splitext(f)[1] == ext]



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
                print(f'Mixture       - Filename: {mix.file_name}, Label: {mix.label}'
                print(f'Vocals        - Filename: {vox.file_name}, Label: {vox.label}'
                print(f'Accompaniment - Filename: {acc.file_name}, Label: {acc.label}'

                # Run an algorithm on the MUSDB18 files and save to disc
                r = nussl.Repet(mix)
                r.run()
                bg_est, fg_est = r.make_audio_signals()
                bg_est.write_audio_to_file(f'{os.path.splitext(mix.file_name)[0]}_bg.wav')
                fg_est.write_audio_to_file(f'{os.path.splitext(mix.file_name)[0]}_fg.wav')

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
                bg_est.write_audio_to_file(f'{os.path.splitext(mix.file_name)[0]}_bg.wav')
                fg_est.write_audio_to_file(f'{os.path.splitext(mix.file_name)[0]}_fg.wav')

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
