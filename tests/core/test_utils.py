import nussl
import numpy as np
from nussl.separation.base import MaskSeparationBase, SeparationBase
from nussl.core.masks import BinaryMask, SoftMask, MaskBase
import pytest

import torch
import random

def test_utils_seed():
    seeds = [0, 123, 666, 15, 2]

    def _get_random():
        r1 = torch.randn(100, 10)
        r2 = np.random.rand(100, 10)
        r3 = random.randint(10, 10000)

        return r1, r2, r3

    for seed in seeds:
        nussl.utils.seed(seed)
        t1 = _get_random()
        nussl.utils.seed(seed)
        t2 = _get_random()

        for first, second in zip(t1, t2):
            assert np.allclose(first, second)

        other_seed = 10

        nussl.utils.seed(other_seed)
        t3 = _get_random()

        for first, second in zip(t1, t3):
            assert not np.allclose(first, second)
            
    # do it again with set_cudnn = True
    for seed in seeds:
        nussl.utils.seed(seed, set_cudnn=True)
        t1 = _get_random()
        nussl.utils.seed(seed, set_cudnn=True)
        t2 = _get_random()

        for first, second in zip(t1, t2):
            assert np.allclose(first, second)

        other_seed = 10

        nussl.utils.seed(other_seed, set_cudnn=True)
        t3 = _get_random()

        for first, second in zip(t1, t3):
            assert not np.allclose(first, second)



def test_utils_find_peak_indices():
    array = np.arange(0, 100)
    peak = nussl.utils.find_peak_indices(array, 1)[0]
    assert peak == 99

    array = np.arange(0, 100).reshape(10, 10)
    peak = nussl.utils.find_peak_indices(array, 3, min_dist=0)
    assert peak == [[9, 9], [9, 8], [9, 7]]

    peak = nussl.utils.find_peak_indices(array, 3, min_dist=(0,))
    assert peak == [[9, 9], [9, 8], [9, 7]]

    peak = nussl.utils.find_peak_indices(array, 3, min_dist=(0, 0))
    assert peak == [[9, 9], [9, 8], [9, 7]]

    peak = nussl.utils.find_peak_indices(array - np.mean(array), 3, min_dist=0)
    assert peak == [[9, 9], [9, 8], [9, 7]]

    pytest.raises(
        ValueError, nussl.utils.find_peak_indices, array, 10, threshold=1.1)

    pytest.warns(
        UserWarning, nussl.utils.find_peak_indices, array, 1000, threshold=1.0)

    pytest.raises(
        ValueError, nussl.utils.find_peak_indices, np.ones((10, 10, 10)), 3, min_dist=0)


def test_utils_complex_randn():
    mat = nussl.utils.complex_randn((100, 100))
    assert (mat.shape == (100, 100))
    assert (mat.dtype == np.complex128)

def test_utils_audio_signal_list(benchmark_audio):
    path = benchmark_audio['dev1_female3_inst_mix.wav']
    signals = [nussl.AudioSignal(path) for i in range(3)]
    assert signals == nussl.utils.verify_audio_signal_list_strict(signals)
    assert signals == nussl.utils.verify_audio_signal_list_lax(signals)
    assert [signals[0]] == nussl.utils.verify_audio_signal_list_lax(signals[0])
    
    dur = signals[0].signal_duration

    signals = [
        nussl.AudioSignal(
            path, 
            duration=np.random.rand() * dur) for i in range(3)
        for i in range(10)
    ]
    pytest.raises(
        ValueError, nussl.utils.verify_audio_signal_list_strict, signals)

    signals = [nussl.AudioSignal(path) for i in range(3)]
    signals[-1].resample(8000)
    pytest.raises(
        ValueError, nussl.utils.verify_audio_signal_list_strict, signals)

    signals = [nussl.AudioSignal(path) for i in range(3)]
    signals[-1].to_mono()
    pytest.raises(
        ValueError, nussl.utils.verify_audio_signal_list_strict, signals)

    signals = [nussl.AudioSignal(path) for i in range(3)]
    signals[-1] = [0, 1, 2]
    pytest.raises(
        ValueError, nussl.utils.verify_audio_signal_list_lax, signals)

    signals = [nussl.AudioSignal(path) for i in range(3)]
    signals[-1] = nussl.AudioSignal()
    pytest.raises(
        ValueError, nussl.utils.verify_audio_signal_list_lax, signals)

    pytest.raises(
        ValueError, nussl.utils.verify_audio_signal_list_lax, {'test': 'garbage'})

def test_utils_audio_signals_to_musdb_track(musdb_tracks):
    track = musdb_tracks[0]
    mixture = nussl.AudioSignal(
        audio_data_array=track.audio,
        sample_rate=track.rate)
    mixture.stft()

    stems = track.stems
    true_sources = {}
    fake_sources = {}
    for k, v in sorted(track.sources.items(), key=lambda x: x[1].stem_id):
        true_sources[k] = nussl.AudioSignal(
            audio_data_array=stems[v.stem_id],
            sample_rate=track.rate
        )
        mask_data = np.random.rand(*mixture.stft_data.shape)
        soft_mask = SoftMask(mask_data)

        _source = mixture.apply_mask(soft_mask)
        _source.istft(truncate_to_length=mixture.signal_length)
        fake_sources[k] = _source

    separated_track = nussl.utils.audio_signals_to_musdb_track(
        mixture, fake_sources, nussl.constants.STEM_TARGET_DICT
    )

    reconstructed_track = nussl.utils.audio_signals_to_musdb_track(
        mixture, true_sources, nussl.constants.STEM_TARGET_DICT
    )

    assert np.allclose(track.stems, reconstructed_track.stems)
    assert track.stems.shape == separated_track.stems.shape

def test_utils_musdb_track_to_audio_signals(musdb_tracks):
    track = musdb_tracks[0]
    stems = track.stems

    mixture, sources = nussl.utils.musdb_track_to_audio_signals(track)

    assert np.allclose(mixture.audio_data, track.audio.T)
    assert mixture.sample_rate == track.rate

    for k, v in sorted(track.sources.items(), key=lambda x: x[1].stem_id):
        assert np.allclose(sources[k].audio_data, stems[v.stem_id].T)
        assert sources[k].sample_rate == track.rate
        assert k in sources[k].path_to_input_file

def test_utils_format():
    _in = '0123~5aBc'
    _gt = '01235abc'
    _est = nussl.utils._format(_in)

    assert _gt == _est

def test_utils_get_axis():
    mat = np.random.rand(100, 10, 1)
    _out = nussl.utils._get_axis(mat, 0, 0)
    assert _out.shape == (10, 1)
    _out = nussl.utils._get_axis(mat, 1, 0)
    assert _out.shape == (100, 1)
    _out = nussl.utils._get_axis(mat, 2, 0)
    assert _out.shape == (100, 10)
