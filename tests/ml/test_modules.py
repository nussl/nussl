import torch
import nussl
from nussl.datasets import transforms
from nussl import ml
import pytest
import numpy as np
import librosa
import itertools


def test_register_module():
    class ExampleModule(torch.nn.Module):
        def forward(self, data):
            data = data * 2
            return data

    assert ExampleModule.__name__ not in (dir(ml.networks.modules))
    ml.register_module(ExampleModule)
    assert ExampleModule.__name__ in (dir(ml.networks.modules))

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
    
def test_shift_and_scale():
    data = torch.randn(100)

    shifter = ml.networks.modules.ShiftAndScale()
    _data = shifter(data)

    assert torch.allclose(data, _data)

    shifter.scale.data[0] = 10
    _data = shifter(data)
    
    assert torch.allclose(10 * data, _data)

    shifter.shift.data[0] = -10
    _data = shifter(data)

    assert torch.allclose(10 * data - 10, _data)

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

def test_ml_group_norm(one_item):
    shape = one_item['mix_magnitude'].shape
    module = ml.networks.modules.GroupNorm(shape[2])
    output = module(one_item['mix_magnitude'])
    assert one_item['mix_magnitude'].shape == output.shape

    module = ml.networks.modules.GroupNorm(shape[2])
    output = module(one_item['mix_magnitude'])
    assert one_item['mix_magnitude'].shape == output.shape

def test_ml_layer_norm(one_item):
    shape = one_item['mix_magnitude'].shape

    for c in range(1, len(shape)):
        dim_combos = list(itertools.combinations(range(len(shape)), c))
        for combo in dim_combos:
            _shape = [shape[x] for x in combo]
            module = ml.networks.modules.LayerNorm(_shape[-1], feature_dims=combo)
            output = module(one_item['mix_magnitude'])
            assert one_item['mix_magnitude'].shape == output.shape

            module = ml.networks.modules.LayerNorm(_shape[0], feature_dims=combo[::-1])
            output = module(one_item['mix_magnitude'])
            assert one_item['mix_magnitude'].shape == output.shape

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

    activations = ['sigmoid', 'tanh', 'relu', 'softmax', ['gated_tanh', 'sigmoid']]
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

            _a = [a, 'l1_norm']

            module = ml.networks.modules.Embedding(
                num_frequencies, 50, e, _a,
                dim_to_embed=-1
            )
            output = module(rnn(data))
            _norm = torch.norm(output, p=1, dim=-1)
            # relu sends entire vectors to zero, so their norm is 0.
            # only check nonzero norm values.
            _norm = _norm[_norm > 0]

            assert torch.allclose(_norm, torch.ones_like(_norm))


def test_ml_mask(one_item):
    data = one_item['mix_magnitude']
    mask = torch.randn(data.shape + (4,))

    masked_data = mask * data.unsqueeze(-1)

    module = ml.networks.modules.Mask()
    output = module(mask, data)

    assert torch.allclose(output, masked_data)

    data = one_item['mix_magnitude']
    ibm = one_item['ideal_binary_mask']

    masked_data = ibm * data.unsqueeze(-1)

    module = ml.networks.modules.Mask()
    output = module(ibm, data)

    assert torch.allclose(output, masked_data)


def test_ml_concatenate(one_item):
    data = one_item['mix_magnitude']
    dims = range(len(data.shape))

    for dim in dims:
        module = ml.networks.modules.Concatenate(dim=dim)
        output = module(data, data)
        assert output.shape[dim] == 2 * data.shape[dim]

def test_ml_split(one_item):
    data = one_item['mix_magnitude']
    dims = range(len(data.shape))

    for dim in dims:
        split_point = np.random.randint(data.shape[dim])
        split_sizes = (split_point, data.shape[dim] - split_point)
        if split_sizes[-1] > 0:
            module = ml.networks.modules.Split(
                split_sizes=split_sizes, dim=dim)
            output = module(data)
            assert len(output) == len(split_sizes)
            for i, o in enumerate(output):
                assert o.shape[dim] == split_sizes[i]

def test_ml_expand():
    tensor_a = torch.rand(100, 10, 5)
    tensor_b = torch.rand(100, 10)

    module = ml.networks.modules.Expand()
    tensor_b = module(tensor_a, tensor_b)

    assert tensor_b.ndim == tensor_a.ndim

    bad_tensor = torch.rand(100, 10, 5, 2)

    pytest.raises(ValueError, module, tensor_a, bad_tensor)


def test_ml_alias():
    modules = {
        'split': {
            'class': 'Split',
            'args': {
                'split_sizes': (3, 7),
                'dim': -1
            }
        },
        'split_zero': {
            'class': 'Alias',
        }
    }

    connections = [
        ('split', ('data',)),
        ('split_zero', ('split:0',))
    ]

    outputs = ['split:0', 'split_zero']

    config = {
        'name': 'AliasModel',
        'modules': modules, 
        'connections': connections,
        'output': outputs
    }

    model = ml.SeparationModel(config)
    data = {'data': torch.randn(100, 10)}
    output = model(data)

    assert 'split_zero' in output
    assert torch.allclose(
        output['split:0'], output['split_zero']
    )

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
        dim = 2 * _product[1] if _product[-3] else _product[1]

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

def test_dual_path(one_item):
    recurrent_stack = {
        'class': 'RecurrentStack',
        'args': {
            'num_features': 100,
            'hidden_size': 50,
            'num_layers': 1,
            'bidirectional': True,
            'dropout': 0.3,
            'rnn_type': 'lstm',
            'batch_first': False
        }
    }
    dual_path = ml.networks.modules.DualPath(
        2, 100, 50, 257, 100, hidden_size=100, 
        intra_processor=recurrent_stack,
        inter_processor=recurrent_stack
    )
    output = dual_path(one_item['mix_magnitude'])
    assert output.shape == one_item['mix_magnitude'].shape

    linear_layer = {
        'class': 'Linear',
        'args': {
            'in_features': 100,
            'out_features': 100,
        }
    }

    dual_path = ml.networks.modules.DualPath(
        2, 100, 50, 257, 100, hidden_size=100, 
        intra_processor=linear_layer,
        inter_processor=linear_layer,
        skip_connection=True
    )
    output = dual_path(one_item['mix_magnitude'])
    assert output.shape == one_item['mix_magnitude'].shape

    nonexisting_layer = {
        'class': 'NoExist',
        'args': {
            'in_features': 100,
            'out_features': 100,
        }
    }

    pytest.raises(ValueError, ml.networks.modules.DualPath,
        2, 100, 50, 257, 100, hidden_size=100, 
        intra_processor=nonexisting_layer,
        inter_processor=nonexisting_layer
    )
