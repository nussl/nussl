import torch
import nussl
from nussl.datasets import transforms
from nussl import ml
import pytest
import numpy as np
import librosa
import itertools

def test_ml_amplitude_to_db(one_item):
    module = ml.networks.modules.AmplitudeToDB()
    output = module(one_item['mix_magnitude'])

    librosa_output = librosa.amplitude_to_db(
        one_item['mix_magnitude'].cpu().numpy()[0],
        amin=1e-4,
        ref=1.0,
    )
    output = output.cpu().numpy()
    assert np.allclose(output, librosa_output)


def test_ml_batch_instance_norm(one_item):    
    module = ml.networks.modules.BatchNorm()
    output = module(one_item['mix_magnitude'])
    assert one_item['mix_magnitude'].shape == output.shape

    module = ml.networks.modules.InstanceNorm(eps=1e-10)
    output = module(one_item['mix_magnitude'])
    assert one_item['mix_magnitude'].shape == output.shape
    _output = output.cpu().numpy()
    assert np.abs(np.mean(_output) - 0.0) < 1e-7
    assert np.abs(np.std(_output) - 1.0) < 1e-3

def test_ml_mel_projection(one_item):
    n_mels = [64, 128, 150]
    data = one_item['mix_magnitude']
    num_frequencies = data.shape[2]

    pytest.raises(ValueError, ml.networks.modules.MelProjection, 
        16000, num_frequencies, 128, direction='neither')

    for n_mel in n_mels:
        forward = ml.networks.modules.MelProjection(
            16000, num_frequencies, n_mel, direction='forward'
        )
        backward = ml.networks.modules.MelProjection(
            16000, num_frequencies, n_mel, direction='backward'
        )
        backward_clamp = ml.networks.modules.MelProjection(
            16000, num_frequencies, n_mel, direction='backward',
            clamp=True
        )

        mel_spec = forward(data).cpu().numpy()[0, :, :, 0].T

        filters = librosa.filters.mel(
            16000, 2 * (num_frequencies - 1), n_mel)
        filters = (
            filters.T / (filters.sum(axis=1) + 1e-8)).T

        assert np.allclose(
            forward.transform.weight.cpu().numpy(),
            filters
        )

        assert np.allclose(
            backward.transform.weight.cpu().numpy(),
            np.linalg.pinv(filters)
        )
        
        _data = data.cpu().numpy()[0, :, :, 0]
        librosa_output = (_data @ filters.T).T

        assert mel_spec.shape[0] == n_mel
        assert np.allclose(mel_spec, librosa_output)

        recon = backward(forward(data))

        _data = data.cpu().numpy()[0, :, :, 0]
        _recon = recon.cpu().numpy()[0, :, :, 0]

        assert np.mean((_data - _recon) ** 2) < 1e-7

        mask = backward_clamp(forward(data)).cpu().numpy()
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

def test_ml_embedding(one_item):
    data = one_item['mix_magnitude']
    num_frequencies = data.shape[2]
    rnn = ml.networks.modules.RecurrentStack(
        num_frequencies, 50, 1, bidirectional=False, dropout=0.0
    )

    
    activations = ['sigmoid', 'tanh', 'relu', 'softmax']
    embedding_sizes = [1, 5, 10, 20, 100]

    for a in activations:
        for e in embedding_sizes:
            module = ml.networks.modules.Embedding(
                num_frequencies, 50, e, a,
                dim_to_embed=-1
            )
            output = module(rnn(data))

            if a == 'sigmoid':
                assert output.min() >= 0
                assert output.max() <= 1
            elif a == 'tanh':
                assert output.min() >= -1
                assert output.max() <= 1
            elif a == 'relu':
                assert output.min() >= 0
            elif a == 'softmax':
                assert output.min() >= 0
                assert output.max() <= 1
                _sum = torch.sum(output, dim=-1)
                assert torch.allclose(_sum, torch.ones_like(_sum))
            
            _a = [a, 'unit_norm']

            module = ml.networks.modules.Embedding(
                num_frequencies, 50, e, _a,
                dim_to_embed=-1
            )
            output = module(rnn(data))
            _norm = torch.norm(output, p=2, dim=-1)
            # relu sends entire vectors to zero, so their norm is 0.
            # only check nonzero norm values.
            _norm = _norm[_norm > 0]

            assert torch.allclose(_norm, torch.ones_like(_norm))

def test_ml_mask(one_item):
    data = one_item['mix_magnitude'].unsqueeze(-1)
    mask = torch.randn(data.shape[:-1] + (4,))

    masked_data = mask * data

    module = ml.networks.modules.Mask()
    output = module(mask, data)

    assert torch.allclose(output, masked_data)

    data = one_item['mix_magnitude'].unsqueeze(-1)
    ibm = one_item['ideal_binary_mask']

    masked_data = ibm * data

    module = ml.networks.modules.Mask()
    output = module(ibm, data)

    assert torch.allclose(output, masked_data)

def test_ml_recurrent_stack(one_item):
    data = one_item['mix_magnitude']
    num_features = data.shape[2]

    pytest.raises(ValueError, ml.networks.modules.RecurrentStack, 
        1, 1, 1, True, .3, 'not_lstm_or_gru'
    )

    rnn_types = ['gru', 'lstm']
    bidirectional = [True, False]
    num_features = [num_features]
    hidden_size = [50, 100]
    num_layers = [1, 2]
    dropout = [.3]

    products = itertools.product(
        num_features, hidden_size, num_layers, bidirectional, dropout, 
        rnn_types)

    for _product in products:
        module = ml.networks.modules.RecurrentStack(*_product)
        output = module(data)
        dim = 2 * _product[1] if _product[-3] else  _product[1]

        assert output.shape == (data.shape[0], data.shape[1], dim)

def test_ml_conv_stack(one_item):
    data = one_item['mix_magnitude']
    num_features = data.shape[2]

    in_channels = 1
    channels = [10, 32, 8, 10]
    dilations = [1, 1, 1, 1]
    filter_shapes = [7, 3, 5, 3]
    residuals = [True, False, False, True]
    batch_norm = True
    use_checkpointing = False

    pytest.raises(ValueError, ml.networks.modules.ConvolutionalStack2D,
        in_channels, [channels[0]], dilations, filter_shapes, residuals,
        batch_norm=batch_norm, use_checkpointing=use_checkpointing
    )

    pytest.warns(UserWarning, ml.networks.modules.ConvolutionalStack2D,
        in_channels, channels, [1, 2, 4, 8], filter_shapes, residuals,
        batch_norm=batch_norm, use_checkpointing=use_checkpointing
    )

    module = ml.networks.modules.ConvolutionalStack2D(
        in_channels, channels, dilations, filter_shapes, residuals,
        batch_norm=batch_norm, use_checkpointing=use_checkpointing
    )
    output = module(data)
    assert (output.shape == (
        data.shape[0], data.shape[1], data.shape[2], channels[-1]))

    module = ml.networks.modules.ConvolutionalStack2D(
        in_channels, channels, dilations, filter_shapes, residuals,
        batch_norm=batch_norm, use_checkpointing=True
    )
    output = module(data)
    assert (output.shape == (
        data.shape[0], data.shape[1], data.shape[2], channels[-1]))
