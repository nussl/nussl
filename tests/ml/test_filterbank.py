from nussl import ml, datasets
from nussl.core.constants import ALL_WINDOWS
import nussl
import pytest
import torch
import itertools
from scipy.signal import check_COLA
import numpy as np

def test_filter_bank(one_item, monkeypatch):
    pytest.raises(
        NotImplementedError, ml.networks.modules.FilterBank, 2048)

    def dummy_filters(self):
        num_filters = (1 + self.filter_length // 2) * 2
        random_basis = torch.randn(
            self.filter_length, num_filters)
        return random_basis.float()

    def dummy_inverse(self):
        num_filters = (1 + self.filter_length // 2) * 2
        random_basis = torch.randn(
            self.filter_length, num_filters)
        return random_basis.float().T

    monkeypatch.setattr(
        ml.networks.modules.FilterBank, 
        'get_transform_filters',
        dummy_filters
    )

    monkeypatch.setattr(
        ml.networks.modules.FilterBank, 
        'get_inverse_filters',
        dummy_inverse
    )

    representation = ml.networks.modules.FilterBank(
        512, hop_length=128)

    data = one_item['mix_audio']

    encoded = representation(data, 'transform')
    decoded = representation(encoded, 'inverse')

    one_sided_shape = list(
        encoded.squeeze(0).shape)
    one_sided_shape[1] = one_sided_shape[1] // 2

    assert tuple(one_sided_shape) == tuple(one_item['mix_magnitude'].shape[1:])
    
    data = one_item['source_audio']

    encoded = representation(data, 'transform')
    decoded = representation(encoded, 'inverse')

    assert decoded.shape == data.shape

def test_filter_bank_alignment(one_item):
    # if we construct a signal with an impulse at a random
    # offset, it should stay in the same place after the
    # stft
    win_length = 256
    hop_length = 64
    win_type = 'sqrt_hann'

    representation = ml.networks.modules.STFT(
        win_length, hop_length=hop_length, window_type=win_type)
    data = torch.zeros_like(one_item['source_audio'])
    for _ in range(20):
        offset = np.random.randint(0, data.shape[-2])
        data[..., offset, 0] = 1

    encoded = representation(data, 'transform')
    decoded = representation(encoded, 'inverse')

    assert torch.allclose(decoded, data, atol=1e-6)

sr = nussl.constants.DEFAULT_SAMPLE_RATE
# Define my window lengths to be powers of 2, ranging from 128 to 2048 samples
win_min = 7  # 2 ** 7  =  128
win_max = 11  # 2 ** 11 = 2048
win_lengths = [2 ** i for i in range(win_min, win_max + 1)]

win_length_32ms = int(2 ** (np.ceil(np.log2(nussl.constants.DEFAULT_WIN_LEN_PARAM * sr))))
win_lengths.append(win_length_32ms)

hop_length_ratios = [0.5, .25]

window_types = ALL_WINDOWS

signals = []

combos = itertools.product(win_lengths, hop_length_ratios, window_types)

@pytest.mark.parametrize("combo", combos)
def test_stft_module(combo, one_item):
    win_length = combo[0]
    hop_length = int(combo[0] * combo[1])
    win_type = combo[2]
    window = nussl.AudioSignal.get_window(combo[2], win_length)
    stft_params = nussl.STFTParams(
        window_length=win_length, hop_length=hop_length, window_type=win_type
    )

    representation = ml.networks.modules.STFT(
        win_length, hop_length=hop_length, window_type=win_type)

    if not check_COLA(window, win_length, win_length - hop_length):
        assert True
    
    data = one_item['mix_audio']

    encoded = representation(data, 'transform')
    decoded = representation(encoded, 'inverse')
    encoded = encoded.squeeze(0).permute(1, 0, 2)

    assert (decoded - data).abs().max() < 1e-5

    audio_signal = nussl.AudioSignal(
        audio_data_array=data.squeeze(0).numpy(), sample_rate=16000, stft_params=stft_params
    )
    nussl_magnitude = np.abs(audio_signal.stft())
    _encoded = encoded.squeeze(0)
    cutoff = _encoded.shape[0] // 2
    _encoded = _encoded[:cutoff, ...]
    assert (_encoded - nussl_magnitude).abs().max() < 1e-6

def test_learned_filterbank(one_item):
    representation = ml.networks.modules.LearnedFilterBank(
        512, hop_length=128, requires_grad=True)

    data = one_item['mix_audio']

    encoded = representation(data, 'transform')
    decoded = representation(encoded, 'inverse')
    
    data = one_item['source_audio']

    encoded = representation(data, 'transform')
    decoded = representation(encoded, 'inverse')

    assert decoded.shape == data.shape
