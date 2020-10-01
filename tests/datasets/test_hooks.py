import pytest
import nussl
from nussl.core import constants
import os
import numpy as np
from nussl.datasets.base_dataset import DataSetException
from nussl.datasets import transforms
import tempfile
import shutil


def test_dataset_hook_musdb18(musdb_tracks):
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

    for k in data['sources']:
        assert k.split('::')[0] in data['metadata']['labels']


def test_dataset_hook_mix_source_folder(mix_source_folder):
    dataset = nussl.datasets.MixSourceFolder(mix_source_folder)
    data = dataset[0]

    _sources = [data['sources'][k] for k in data['sources']]
    assert np.allclose(sum(_sources).audio_data, data['mix'].audio_data)

    for k in data['sources']:
        assert k.split('::')[0] in data['metadata']['labels']

    dataset = nussl.datasets.MixSourceFolder(mix_source_folder, make_mix=True)
    data = dataset[0]

    _sources = [data['sources'][k] for k in data['sources']]
    assert np.allclose(sum(_sources).audio_data, data['mix'].audio_data)

    for k in data['sources']:
        assert k.split('::')[0] in data['metadata']['labels']


def test_dataset_hook_scaper_folder(scaper_folder):
    dataset = nussl.datasets.Scaper(scaper_folder)
    data = dataset[0]

    _sources = [data['sources'][k] for k in data['sources']]
    assert np.allclose(sum(_sources).audio_data, data['mix'].audio_data)

    for k in data['sources']:
        assert k.split('::')[0] in data['metadata']['labels']

    # make sure SumSources transform works
    tfm = transforms.SumSources(
        [['050', '051']],
        group_names=['both'],
    )

    data = tfm(data)

    for k in data['sources']:
        assert k.split('::')[0] in data['metadata']['labels']

    _sources = [data['sources'][k] for k in data['sources']]
    assert np.allclose(sum(_sources).audio_data, data['mix'].audio_data)


def test_dataset_hook_bad_scaper_folder(bad_scaper_folder):
    pytest.raises(
        DataSetException, nussl.datasets.Scaper, bad_scaper_folder)


def test_dataset_hook_wham(benchmark_audio):
    # make a fake wham dir structure
    audio = nussl.AudioSignal(benchmark_audio['K0140.wav'])
    with tempfile.TemporaryDirectory() as tmpdir:
        for wav_folder in ['wav8k', 'wav16k']:
            sr = 8000 if wav_folder == 'wav8k' else 16000
            for mode in ['min', 'max']:
                for split in ['tr', 'cv', 'tt']:
                    for key, val in nussl.datasets.WHAM.MIX_TO_SOURCE_MAP.items():
                        parent = os.path.join(
                            tmpdir, wav_folder, mode, split)
                        mix_path = os.path.join(parent, key)
                        os.makedirs(mix_path, exist_ok=True)
                        audio.resample(sr)
                        audio.write_audio_to_file(
                            os.path.join(mix_path, '0.wav'))
                        for x in val:
                            source_path = os.path.join(parent, x)
                            os.makedirs(source_path, exist_ok=True)
                            audio.write_audio_to_file(
                                os.path.join(source_path, '0.wav'))

        for mode in ['min', 'max']:
            for split in ['tr', 'cv', 'tt']:
                for sr in [8000, 16000]:
                    wham = nussl.datasets.WHAM(
                        root=tmpdir, mix_folder='mix_clean', mode=mode,
                        split=split, sample_rate=sr)
                    output = wham[0]
                    assert output['metadata']['labels'] == ['s1', 's2']

                    wham = nussl.datasets.WHAM(
                        root=tmpdir, mix_folder='mix_both', mode=mode,
                        split=split, sample_rate=sr)
                    output = wham[0]
                    assert output['metadata']['labels'] == ['s1', 's2', 'noise']

                    wham = nussl.datasets.WHAM(
                        root=tmpdir, mix_folder='mix_single', mode=mode,
                        split=split, sample_rate=sr)
                    output = wham[0]
                    assert output['metadata']['labels'] == ['s1']

        pytest.raises(DataSetException, nussl.datasets.WHAM,
                      tmpdir, mix_folder='not matching')

        pytest.raises(DataSetException, nussl.datasets.WHAM,
                      tmpdir, mode='not matching')

        pytest.raises(DataSetException, nussl.datasets.WHAM,
                      tmpdir, split='not matching')

        pytest.raises(DataSetException, nussl.datasets.WHAM,
                      tmpdir, sample_rate=44100)

def test_dataset_hook_fuss(scaper_folder):
    pytest.raises(DataSetException, nussl.datasets.FUSS, 'folder', 
        split='bad split')

    with tempfile.TemporaryDirectory() as tmpdir:
        train_folder = os.path.join(tmpdir, 'train')
        shutil.copytree(scaper_folder, train_folder)

        fuss = nussl.datasets.FUSS(tmpdir)

def test_dataset_hook_on_the_fly():
    def make_sine_wave(freq, sample_rate, duration):
        dt = 1 / sample_rate
        x = np.arange(0.0, duration, dt)
        x = np.sin(2 * np.pi * freq * x)
        return x

    n_sources = 2
    duration = 3
    sample_rate = 44100
    min_freq, max_freq = 110, 1000
    def make_mix(dataset, i):
        sources = {}
        freqs = []
        for i in range(n_sources):
            freq = np.random.randint(min_freq, max_freq)
            freqs.append(freq)
            source_data = make_sine_wave(freq, sample_rate, duration)
            source_signal = dataset._load_audio_from_array(
                audio_data=source_data, sample_rate=sample_rate)
            sources[f'sine{i}'] = source_signal * 1 / n_sources
        mix = sum(sources.values())
        output = {
            'mix': mix,
            'sources': sources,
            'metadata': {
                'frequencies': freqs    
            }    
        }
        return output
    dataset = nussl.datasets.OnTheFly(make_mix, 10)
    assert len(dataset) == 10

    for output in dataset:
        assert output['mix'] == sum(output['sources'].values())
        assert output['mix'].signal_duration == duration

    def bad_mix_closure(dataset, i):
        return 'not a dictionary'
    pytest.raises(DataSetException, nussl.datasets.OnTheFly, bad_mix_closure, 10)

    def bad_dict_closure(dataset, i):
        return {'key': 'no mix in this dict'}
    pytest.raises(DataSetException, nussl.datasets.OnTheFly, bad_dict_closure, 10)

    def bad_dict_sources_closure(dataset, i):
        return {'mix': 'no sources in this dict'}
    pytest.raises(DataSetException, nussl.datasets.OnTheFly, bad_dict_sources_closure, 10)
