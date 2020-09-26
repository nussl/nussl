import pytest
from nussl.datasets import BaseDataset, transforms
from nussl.datasets.base_dataset import DataSetException
import nussl
from nussl import STFTParams
import numpy as np
import soundfile as sf
import itertools
import tempfile
import os
import torch


class BadTransform(object):
    def __init__(self, fake=None):
        self.fake = fake

    def __call__(self, data):
        return 'not a dictionary'


class BadDataset(BaseDataset):
    def get_items(self, folder):
        return {'anything': 'not a list'}

    def process_item(self, item):
        return 'not a dictionary'


def dummy_process_item(self, item):
    audio = self._load_audio_file(item)
    output = {
        'mix': audio,
        'sources': {'key': audio}
    }
    return output


def dummy_process_item_by_audio(self, item):
    data, sr = sf.read(item)
    audio = self._load_audio_from_array(data, sr)
    output = {
        'mix': audio,
        'sources': {'key': audio}
    }
    return output


def initialize_bad_dataset_and_run():
    _bad_transform = BadTransform()
    _bad_dataset = BadDataset('test', transform=_bad_transform)
    _ = _bad_dataset[0]


def test_dataset_base(benchmark_audio, monkeypatch):
    keys = [benchmark_audio[k] for k in benchmark_audio]

    def dummy_get(self, folder):
        return keys

    pytest.raises(DataSetException, initialize_bad_dataset_and_run)

    monkeypatch.setattr(BadDataset, 'get_items', dummy_get)
    pytest.raises(DataSetException, initialize_bad_dataset_and_run)

    monkeypatch.setattr(BadDataset, 'process_item', dummy_process_item)
    pytest.raises(transforms.TransformException, initialize_bad_dataset_and_run)

    monkeypatch.setattr(BaseDataset, 'get_items', dummy_get)
    monkeypatch.setattr(BaseDataset, 'process_item', dummy_process_item)

    _dataset = BaseDataset('test')

    assert len(_dataset) == len(keys)

    audio_signal = nussl.AudioSignal(keys[0])
    assert _dataset[0]['mix'] == audio_signal

    _dataset = BaseDataset('test', transform=BadTransform())
    pytest.raises(transforms.TransformException, _dataset.__getitem__, 0)

    psa = transforms.MagnitudeSpectrumApproximation()
    _dataset = BaseDataset('test', transform=psa)

    output = _dataset[0]
    assert 'source_magnitudes' in output
    assert 'mix_magnitude' in output
    assert 'ideal_binary_mask' in output

    monkeypatch.setattr(
        BaseDataset, 'process_item', dummy_process_item_by_audio)
    psa = transforms.MagnitudeSpectrumApproximation()
    _dataset = BaseDataset('test', transform=psa)

    output = _dataset[0]
    assert 'source_magnitudes' in output
    assert 'mix_magnitude' in output
    assert 'ideal_binary_mask' in output

    _dataset.transform = transforms.Compose([
        transforms.MagnitudeSpectrumApproximation(),
        transforms.ToSeparationModel()
    ])

    dataloader = torch.utils.data.DataLoader(_dataset, shuffle=False, num_workers=8)
    assert len(list(dataloader)) == len(_dataset)
    for idx, batch in enumerate(dataloader):
        assert torch.allclose(batch['mix_magnitude'][0], _dataset[idx]['mix_magnitude'])


def test_dataset_base_filter(benchmark_audio, monkeypatch):
    keys = [benchmark_audio[k] for k in benchmark_audio]

    def dummy_get(self, folder):
        return keys

    monkeypatch.setattr(BaseDataset, 'get_items', dummy_get)
    monkeypatch.setattr(BaseDataset, 'process_item', dummy_process_item)

    _dataset = BaseDataset('test')
    min_length = 7 # in seconds

    # self here refers to the dataset
    def remove_short_audio(self, item):
        processed_item = self.process_item(item)
        mix_length = processed_item['mix'].signal_duration
        if mix_length < min_length:
            return False
        return True
    
    _dataset.filter_items_by_condition(remove_short_audio)
    for item in _dataset:
        assert item['mix'].signal_duration >= min_length

    def bad_filter_func(self, item):
        return 'not a bool!'
    
    pytest.raises(
        DataSetException, _dataset.filter_items_by_condition, bad_filter_func)

def test_dataset_base_audio_signal_params(benchmark_audio, monkeypatch):
    keys = [benchmark_audio[k] for k in benchmark_audio]

    def dummy_get(self, folder):
        return keys

    monkeypatch.setattr(BaseDataset, 'get_items', dummy_get)

    monkeypatch.setattr(
        BaseDataset, 'process_item', dummy_process_item_by_audio)

    stft_params = [
        STFTParams(
            window_length=256,
            hop_length=32,
            window_type='triang'),
        None
    ]

    sample_rates = [4000, None]
    num_channels = [1, 2, None]
    strict_sample_rate = [False, True]

    product = itertools.product(
        stft_params, sample_rates, num_channels, strict_sample_rate)

    def _get_outputs(dset):
        outputs = []
        for i in range(len(dset)):
            outputs.append(dset[i])
        return outputs

    for s, sr, nc, s_sr in product:
        if s_sr and sr is not None:
            pytest.raises(
                DataSetException, BaseDataset, 'test', stft_params=s,
                sample_rate=sr, num_channels=nc, strict_sample_rate=s_sr)
            continue

        _dataset = BaseDataset(
            'test', stft_params=s,
            sample_rate=sr, num_channels=nc,
            strict_sample_rate=s_sr)
        outputs = _get_outputs(_dataset)

        # they should all have the same sample rate and stft
        _srs = []
        _stfts = []

        for i, o in enumerate(outputs):
            if sr:
                assert o['mix'].sample_rate == sr
            if s:
                assert o['mix'].stft_params == s
            if nc:
                if o['mix'].num_channels < nc:
                    assert pytest.warns(UserWarning, _dataset.__getitem__, i)
                else:
                    assert o['mix'].num_channels == nc
            _srs.append(o['mix'].sample_rate)
            _stfts.append(o['mix'].stft_params)

        for _sr, _stft in zip(_srs, _stfts):
            assert _sr == _srs[0]
            assert _stft == _stfts[0]


def test_dataset_base_with_caching(benchmark_audio, monkeypatch):
    keys = [benchmark_audio[k] for k in benchmark_audio]

    def dummy_get(self, folder):
        return keys

    monkeypatch.setattr(BaseDataset, 'get_items', dummy_get)
    monkeypatch.setattr(
        BaseDataset, 'process_item', dummy_process_item_by_audio)

    with tempfile.TemporaryDirectory() as tmpdir:
        tfm = transforms.Cache(
            os.path.join(tmpdir, 'cache'), overwrite=True)

        _dataset = BaseDataset('test', transform=tfm, cache_populated=False)
        assert tfm.cache_size == len(_dataset)

        _data_a = _dataset[0]
        _dataset.cache_populated = True
        pytest.raises(transforms.TransformException,
                      _dataset.__getitem__, 1)  # haven't written to this yet!
        assert len(_dataset.post_cache_transforms.transforms) == 1
        _data_b = _dataset[0]

        for key in _data_a:
            assert _data_a[key] == _data_b[key]

        _dataset.cache_populated = False

        outputs_a = []
        outputs_b = []

        for i in range(len(_dataset)):
            outputs_a.append(_dataset[i])

        _dataset.cache_populated = True

        for i in range(len(_dataset)):
            outputs_b.append(_dataset[i])

        for _data_a, _data_b in zip(outputs_a, outputs_b):
            for key in _data_a:
                assert _data_a[key] == _data_b[key]

    with tempfile.TemporaryDirectory() as tmpdir:
        tfm = transforms.Compose([
            transforms.MagnitudeSpectrumApproximation(),
            transforms.ToSeparationModel(),
            transforms.Cache(
                os.path.join(tmpdir, 'cache'), overwrite=True),
        ])
        _dataset = BaseDataset('test', transform=tfm, cache_populated=False)
        assert tfm.transforms[-1].cache_size == len(_dataset)
        _data_a = _dataset[0]

        _dataset.cache_populated = True
        pytest.raises(transforms.TransformException,
                      _dataset.__getitem__, 1)  # haven't written to this yet!
        assert len(_dataset.post_cache_transforms.transforms) == 1
        _data_b = _dataset[0]

        for key in _data_a:
            if torch.is_tensor(_data_a[key]):
                assert torch.allclose(_data_a[key], _data_b[key])
            else:
                assert _data_a[key] == _data_b[key]

        _dataset.cache_populated = False

        outputs_a = []
        outputs_b = []

        for i in range(len(_dataset)):
            outputs_a.append(_dataset[i])

        _dataset.cache_populated = True

        for i in range(len(_dataset)):
            outputs_b.append(_dataset[i])

        for _data_a, _data_b in zip(outputs_a, outputs_b):
            for key in _data_a:
                if torch.is_tensor(_data_a[key]):
                    assert torch.allclose(_data_a[key], _data_b[key])
                else:
                    assert _data_a[key] == _data_b[key]

    for L in [100, 400, 1000]:
        with tempfile.TemporaryDirectory() as tmpdir:
            tfm = transforms.Compose([
                transforms.MagnitudeSpectrumApproximation(),
                transforms.ToSeparationModel(),
                transforms.Cache(
                    os.path.join(tmpdir, 'cache'), overwrite=True),
                transforms.GetExcerpt(L)
            ])
            _dataset = BaseDataset('test', transform=tfm, cache_populated=False)
            assert tfm.transforms[-2].cache_size == len(_dataset)
            assert len(_dataset.post_cache_transforms.transforms) == 2

            for i in range(len(_dataset)):
                _ = _dataset[i]

            _dataset.cache_populated = True
            outputs = []
            for i in range(len(_dataset)):
                outputs.append(_dataset[i])

            for _output in outputs:
                for key, val in _output.items():
                    if torch.is_tensor(val):
                        assert val.shape[0] == L
