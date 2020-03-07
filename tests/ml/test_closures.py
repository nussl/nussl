from nussl import datasets, ml
from torch import optim, nn
import torch
import numpy as np
from nussl.ml.train.closures import ClosureException
import pytest

def test_base_closure():
    n_batch = 40
    n_time = 400
    n_freq = 129
    n_sources = 4
    n_embedding = 20

    embedding = torch.rand(n_batch, n_time, n_freq, n_embedding)
    embedding = torch.nn.functional.normalize(
        embedding, dim=-1, p=2)

    assignments = torch.rand(n_batch, n_time, n_freq, n_sources) > .5
    assignments = assignments.float()

    weights = torch.ones(n_batch, n_time, n_freq)

    output = {
        'embedding': embedding,
        'weights': weights,
        'estimates': torch.rand(n_batch, n_time, n_freq, n_sources)
    }

    target = {
        'ideal_binary_mask': assignments,
        'source_magnitudes': torch.rand(n_batch, n_time, n_freq, n_sources)
    }

    loss_dictionary = {
        'DeepClusteringLoss': {
            'weight': .2,
        },
        'L1Loss': {
            'weight': .8
        }
    }

    closure = ml.train.closures.Closure(loss_dictionary)
    loss_a = closure.compute_loss(output, target)
    weighted_sum = 0
    
    for key in loss_dictionary:
        assert key in loss_a
        weighted_sum += loss_a[key].item() * loss_dictionary[key]['weight']

    assert 'loss' in loss_a
    assert np.allclose(loss_a['loss'].item(), weighted_sum, atol=1e-4)

    loss_dictionary = {
        'DeepClusteringLoss': {
            'weight': .2,
        },
        'PermutationInvariantLoss': {
            'weight': .8,
            'args': ['L1Loss']
        }
    }

    # checking validation
    pytest.raises(ClosureException, ml.train.closures.Closure, ['not a dict'])
    pytest.raises(ClosureException, ml.train.closures.Closure, {'no_matching_loss': {}})
    pytest.raises(ClosureException, ml.train.closures.Closure, 
        {
            'DeepClusteringLoss': ['not a dict']
        }
    )
    pytest.raises(ClosureException, ml.train.closures.Closure, 
        {
            'DeepClusteringLoss': {
                'bad_val_key': []
            }
        }
    )
    pytest.raises(ClosureException, ml.train.closures.Closure, 
        {
            'DeepClusteringLoss': {
                'weight': 'not a float or int'
            }
        }
    )
    pytest.raises(ClosureException, ml.train.closures.Closure, 
        {
            'DeepClusteringLoss': {
                'weight': 1,
                'args': {'not a list': 'woo'}
            }
        }
    )
    pytest.raises(ClosureException, ml.train.closures.Closure, 
        {
            'DeepClusteringLoss': {
                'weight': 1,
                'args': [],
                'kwargs': ['not a dict']
            }
        }
    ) 

    closure = ml.train.closures.Closure(loss_dictionary)
    loss_b = closure.compute_loss(output, target)
    weighted_sum = 0
    
    for key in loss_dictionary:
        assert key in loss_b
        weighted_sum += loss_b[key].item() * loss_dictionary[key]['weight']

    assert 'loss' in loss_b
    assert np.allclose(loss_b['loss'].item(), weighted_sum, atol=1e-4)

    assert np.allclose(loss_a['loss'].item(), loss_b['loss'].item(), atol=1e-2)

    class CustomLoss:
        DEFAULT_KEYS = {}
        pass

    custom_loss_dictionary = {
        'CustomLoss': {
            'weight': .8,
        }
    }

    closure = ml.train.closures.Closure(
        custom_loss_dictionary, custom_losses=[CustomLoss])
    assert isinstance(closure.losses[0][0], CustomLoss)
    

def test_train_and_validate_closure():
    n_batch = 5
    n_time = 100
    n_freq = 129
    n_sources = 2
    n_embedding = 20

    chimera_config = ml.networks.builders.build_recurrent_chimera(
        n_freq, 50, 2, True, 0.3, 20, ['sigmoid', 'unit_norm'], 
        2, 'softmax', 
    )
    model = ml.networks.SeparationModel(chimera_config)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    assignments = torch.rand(n_batch, n_time, n_freq, n_sources) > .5
    assignments = assignments.float()
    weights = torch.ones(n_batch, n_time, n_freq)

    data = {
        'mix_magnitude': torch.rand(n_batch, n_time, n_freq).unsqueeze(-1),
        'ideal_binary_mask': assignments,
        'weights': weights,
        'source_magnitudes': torch.rand(n_batch, n_time, n_freq, n_sources)
    }

    loss_dictionary = {
        'DeepClusteringLoss': {
            'weight': .2,
        },
        'L1Loss': {
            'weight': .8
        }
    }

    closure = ml.train.closures.TrainClosure(
        loss_dictionary, optimizer, model)
    
    # need a hack here to put "None" since closure expects an
    # engine within an ignite object, but we're not using
    # ignite in the tests
    init_loss = closure(None, data)

    for i in range(100):
        loss = closure(None, data)
    
    last_loss = loss

    for key in last_loss:
        assert last_loss[key] < init_loss[key]

    closure = ml.train.closures.ValidationClosure(
        loss_dictionary, model)

    for i in range(1):
        loss = closure(None, data)

    for key in loss:
        assert np.allclose(loss[key], last_loss[key], 1e-1)
