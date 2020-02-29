import pytest
import nussl
from nussl.core import constants
import os
import numpy as np
from nussl.datasets.base_dataset import DataSetException

def test_datasets_musdb18(musdb_tracks):
    dataset = nussl.datasets.MUSDB18(
        folder=musdb_tracks.root, download=True)

    data = dataset[0]
    track = musdb_tracks[0]
    stems = track.stems

    assert np.allclose(data['mix'].audio_data, track.audio.T)

    for k, v in sorted(track.sources.items(), key=lambda x: x[1].stem_id):
        assert np.allclose(
            data['sources'][k].audio_data, stems[v.stem_id].T)
    
    dataset = nussl.datasets.MUSDB18(
        folder=None, download=False)
    assert dataset.folder == os.path.join(
        constants.DEFAULT_DOWNLOAD_DIRECTORY, 'musdb18')

def test_datasets_mix_source_folder(mix_source_folder):
    dataset = nussl.datasets.MixSourceFolder(mix_source_folder)
    data = dataset[0]

    _sources = [data['sources'][k] for k in data['sources']]
    assert np.allclose(sum(_sources).audio_data, data['mix'].audio_data)

def test_datasets_scaper_folder(scaper_folder):
    dataset = nussl.datasets.Scaper(scaper_folder)
    data = dataset[0]

    _sources = [data['sources'][k] for k in data['sources']]
    assert np.allclose(sum(_sources).audio_data, data['mix'].audio_data)

def test_datasets_bad_scaper_folder(bad_scaper_folder):
    dataset = nussl.datasets.Scaper(bad_scaper_folder)
    pytest.raises(DataSetException, dataset.__getitem__, 0)
        