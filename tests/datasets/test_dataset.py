import pytest
from nussl.datasets import BaseDataset, transforms
from nussl.datasets.base_dataset import DataSetException
import nussl
from nussl import STFTParams
import numpy as np

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

def test_dataset_audio_signal(benchmark_audio, musdb_tracks, monkeypatch):
    keys = [benchmark_audio[k] for k in benchmark_audio]        

    def dummy_get(self, folder):
        return keys

    def dummy_process_item(self, item):
        audio = self._load_audio_file(item)
        output = {
            'mix': audio
        }
        return output

    def initialize_bad_dataset_and_run():
        _bad_transform = BadTransform()
        _bad_dataset = BadDataset('test', transforms=[_bad_transform])
        _bad_dataset[0]

    pytest.raises(DataSetException, initialize_bad_dataset_and_run)

    monkeypatch.setattr(BadDataset, 'get_items', dummy_get)
    pytest.raises(DataSetException, initialize_bad_dataset_and_run)

    monkeypatch.setattr(BadDataset, 'process_item', dummy_process_item)
    pytest.raises(DataSetException, initialize_bad_dataset_and_run)

    monkeypatch.setattr(BaseDataset, 'get_items', dummy_get)
    monkeypatch.setattr(BaseDataset, 'process_item', dummy_process_item)

    _dataset = BaseDataset('test')

    assert len(_dataset) == len(keys)

    audio_signal = nussl.AudioSignal(keys[0])
    assert _dataset[0]['mix'] == audio_signal
