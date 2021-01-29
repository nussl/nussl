import nussl
import numpy as np
from nussl.separation.base import MaskSeparationBase, SeparationBase
from nussl.core.masks import BinaryMask, SoftMask, MaskBase
import pytest

import torch
import random
import matplotlib.pyplot as plt
import os
import tempfile

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


def test_utils_slice_along_dim():
    data = [
        np.random.rand(10, 10, 10, 10, 10),
        torch.rand(10, 10, 10, 10, 10)
    ]
    for _data in data:
        dims = range(len(_data.shape))

        for d in dims:
            _first = np.random.randint(_data.shape[d])
            _second = np.random.randint(_data.shape[d])
            start = min(_first, _second)
            end = max(_first, _second)

            if d > 3:
                pytest.raises(ValueError,
                              nussl.utils._slice_along_dim,
                              _data, d, start, end)
            else:
                sliced_data = nussl.utils._slice_along_dim(
                    _data, d, start, end)

                expected_shape = list(_data.shape)
                expected_shape[d] = end - start
                expected_shape = tuple(expected_shape)

                assert sliced_data.shape == expected_shape

    data = np.random.rand(10, 10)
    pytest.raises(ValueError, nussl.utils._slice_along_dim,
                  data, 2, 0, 10)

PLOT_DIRECTORY = 'tests/utils/plots'
os.makedirs(PLOT_DIRECTORY, exist_ok=True)

def test_utils_visualize_spectrogram(music_mix_and_sources):
    mix, sources = music_mix_and_sources

    plt.figure(figsize=(10, 9))
    plt.subplot(211)
    nussl.utils.visualize_spectrogram(mix)

    plt.subplot(212)
    nussl.utils.visualize_spectrogram(mix, do_mono=True)

    plt.subplot(313)
    nussl.utils.visualize_spectrogram(mix, y_axis='mel')

    OUTPUT = os.path.join(PLOT_DIRECTORY, 'viz_spectrogram.png')
    plt.tight_layout()
    plt.savefig(OUTPUT)

def test_utils_visualize_waveplot(music_mix_and_sources):
    mix, sources = music_mix_and_sources

    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    nussl.utils.visualize_waveform(mix)

    plt.subplot(212)
    nussl.utils.visualize_waveform(mix, do_mono=True)

    OUTPUT = os.path.join(PLOT_DIRECTORY, 'viz_waveform.png')
    plt.tight_layout()
    plt.savefig(OUTPUT)


def test_utils_visualize_sources(music_mix_and_sources):
    mix, sources = music_mix_and_sources
    colors = None

    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    nussl.utils.visualize_sources_as_masks(
        sources, db_cutoff=-70, alpha_amount=2.0,
        y_axis='mel', colors=colors)
    plt.subplot(212)
    nussl.utils.visualize_sources_as_waveform(
        sources, colors=colors, show_legend=True)

    OUTPUT = os.path.join(PLOT_DIRECTORY, 'viz_sources_dict.png')
    plt.tight_layout()
    plt.savefig(OUTPUT)

    sources = list(sources.values())
    colors = ['blue', 'red']

    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    nussl.utils.visualize_sources_as_masks(
        sources, db_cutoff=-70, alpha_amount=2.0,
        y_axis='mel', do_mono=True, colors=colors)
    plt.subplot(212)
    nussl.utils.visualize_sources_as_waveform(
        sources, do_mono=True, colors=colors, 
        show_legend=False)

    OUTPUT = os.path.join(PLOT_DIRECTORY, 'viz_sources_list.png')
    plt.tight_layout()
    plt.savefig(OUTPUT)

def test_close_temp_files():
    '''
    Create a bunch of temp files and then make sure they've been closed and
    deleted. This test is taken wholesale from Scaper.
    '''
    # With delete=True
    tmpfiles = []
    with nussl.utils._close_temp_files(tmpfiles):
        for _ in range(5):
            tmpfiles.append(
                tempfile.NamedTemporaryFile(suffix='.wav', delete=True))

    for tf in tmpfiles:
        assert tf.file.closed
        assert not os.path.isfile(tf.name)

    # With delete=False
    tmpfiles = []
    with nussl.utils._close_temp_files(tmpfiles):
        for _ in range(5):
            tmpfiles.append(
                tempfile.NamedTemporaryFile(suffix='.wav', delete=False))

    for tf in tmpfiles:
        assert tf.file.closed
        assert not os.path.isfile(tf.name)

    # with an exception before exiting
    try:
        tmpfiles = []
        with nussl.utils._close_temp_files(tmpfiles):
            tmpfiles.append(
                tempfile.NamedTemporaryFile(suffix='.wav', delete=True))
            raise ValueError
    except ValueError:
        for tf in tmpfiles:
            assert tf.file.closed
            assert not os.path.isfile(tf.name)
    else:
        assert False, 'Exception was not reraised.'
