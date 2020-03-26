import nussl
import os
import tempfile
import pytest
from nussl.core.efz_utils import (
    NoConnectivityError, FailedDownloadError,
    MismatchedHashError, MetadataError
)
from nussl.core import constants
from random import shuffle
import numpy as np
from six.moves.urllib_parse import urljoin


def get_smallest_file(available_files):
    smallest = np.inf
    best = 0
    for i, a in enumerate(available_files):
        if int(a['file_size_bytes']) < smallest:
            smallest = int(a['file_size_bytes'])
            best = i
    return best


def test_efz_download_audio():
    available_audio_files = nussl.efz_utils.get_available_audio_files()
    best = get_smallest_file(available_audio_files)
    key = available_audio_files[best]['file_name']

    with tempfile.TemporaryDirectory() as tmp_dir:
        path1 = nussl.efz_utils.download_audio_file(key)
        assert os.path.exists(path1)
        assert os.path.expanduser('~/.nussl/') in path1

        path2 = nussl.efz_utils.download_audio_file(key, tmp_dir)
        assert os.path.exists(path2)

        current_hash = nussl.efz_utils._hash_file(path2)
        a = nussl.AudioSignal(path2, sample_rate=8000)
        a.write_audio_to_file(path2)
        next_hash = nussl.efz_utils._hash_file(path2)

        assert current_hash != next_hash
        path2 = nussl.efz_utils.download_audio_file(key, tmp_dir)
        next_hash = nussl.efz_utils._hash_file(path2)
        assert current_hash == next_hash

        a.write_audio_to_file(path2 + 'garbage')
        pytest.raises(FailedDownloadError, nussl.core.efz_utils._download_file,
                      key + 'garbage', constants.NUSSL_EFZ_AUDIO_URL, tmp_dir, 'audio')
        assert not os.path.exists(path2 + 'garbage')

        pytest.raises(FailedDownloadError, nussl.core.efz_utils._download_file,
                      key, 'garbage' + constants.NUSSL_EFZ_AUDIO_URL, tmp_dir, 'audio')

        file_url = urljoin(constants.NUSSL_EFZ_AUDIO_URL, key)

        pytest.raises(MismatchedHashError, nussl.core.efz_utils._download_file,
                      key, file_url, tmp_dir, 'audio', file_hash=123)


def test_efz_download_benchmark():
    available_benchmark_files = nussl.efz_utils.get_available_benchmark_files()
    best = get_smallest_file(available_benchmark_files)
    key = available_benchmark_files[best]['file_name']

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = nussl.efz_utils.download_benchmark_file(key, tmp_dir)
        assert os.path.exists(path)
        _hash = nussl.efz_utils._hash_file(path)
        assert _hash == available_benchmark_files[best]['file_hash']


def test_efz_download_trained_model():
    available_model_files = nussl.efz_utils.get_available_trained_models()
    best = get_smallest_file(available_model_files)
    key = available_model_files[best]['file_name']

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = nussl.efz_utils.download_trained_model(key, tmp_dir)
        assert os.path.exists(path)
        _hash = nussl.efz_utils._hash_file(path)
        assert _hash == available_model_files[best]['file_hash']


def test_efz_exceptions():
    available_model_files = nussl.efz_utils.get_available_trained_models()
    key = available_model_files[0]['file_name']

    pytest.raises(MetadataError, nussl.efz_utils._download_metadata_for_file,
                  key, 'audio')
    pytest.raises(MetadataError, nussl.efz_utils._download_metadata_for_file,
                  key, 'garbage')

    pytest.raises(NoConnectivityError, nussl.efz_utils._download_all_metadata,
                  constants.NUSSL_EFZ_BASE_URL + 'garbage')


def test_hashing():
    first_hash = nussl.efz_utils._hash_directory('.')
    second_hash = nussl.efz_utils._hash_directory('.')

    assert first_hash == second_hash

    first_hash = nussl.efz_utils._hash_directory('.', ext='.py')
    second_hash = nussl.efz_utils._hash_directory('.', ext='.py')

    assert first_hash == second_hash


def test_efz_show_available():
    nussl.efz_utils.print_available_trained_models()
    nussl.efz_utils.print_available_audio_files()
    nussl.efz_utils.print_available_benchmark_files()
