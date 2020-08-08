from nussl import datasets, ml
from torch import optim, nn
import torch
import numpy as np
from nussl.ml.train.closures import ClosureException
import pytest
import nussl


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

    closure = ml.train.closures.Closure(
        loss_dictionary, combination_approach='combine_by_multiply')
    loss_a = closure.compute_loss(output, target)
    weighted_product = 1

    for key in loss_dictionary:
        assert key in loss_a
        weighted_product *= loss_a[key].item() * loss_dictionary[key]['weight']

    assert 'loss' in loss_a
    assert np.allclose(loss_a['loss'].item(), weighted_product, atol=1e-4)

    loss_dictionary = {
        'DeepClusteringLoss': {
            'weight': .2,
        },
        'PermutationInvariantLoss': {
            'weight': .8,
            'args': ['L1Loss']
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
    # doing it twice should work
    closure = ml.train.closures.Closure(loss_dictionary)
    loss_b = closure.compute_loss(output, target)
    weighted_sum = 0

    for key in loss_dictionary:
        assert key in loss_b
        weighted_sum += loss_b[key].item() * loss_dictionary[key]['weight']

    assert 'loss' in loss_b
    assert np.allclose(loss_b['loss'].item(), weighted_sum, atol=1e-4)

    assert np.allclose(loss_a['loss'].item(), loss_b['loss'].item(), atol=1e-2)

    loss_dictionary = {
        'PITLoss': {
            'class': 'PermutationInvariantLoss',
            'keys': {'audio': 'estimates', 'source_audio': 'targets'},
            'args': [{
                'class': 'SISDRLoss',
                'kwargs': {'scaling': False}
            }]
        }
    }

    closure = ml.train.closures.Closure(loss_dictionary)
    # doing it twice should work
    closure = ml.train.closures.Closure(loss_dictionary)
    audio = torch.rand(n_batch, 44100, 2)
    source_audio = torch.rand(n_batch, 44100, 2)

    output = {
        'audio': audio,
    }

    target = {
        'source_audio': source_audio,
    }

    loss_b = closure.compute_loss(output, target)

    class CustomLoss:
        DEFAULT_KEYS = {}
        pass

    custom_loss_dictionary = {
        'CustomLoss': {
            'weight': .8,
        }
    }

    ml.register_loss(CustomLoss)

    closure = ml.train.closures.Closure(custom_loss_dictionary)
    assert isinstance(closure.losses[0][0], CustomLoss)

def test_multitask_combination():
    nussl.utils.seed(0)

    n_batch = 40
    n_time = 400
    n_freq = 129
    n_sources = 4

    output = {
        'estimates_a': 3 * torch.rand(n_batch, n_time, n_freq, n_sources),
        'estimates_b': .1 * torch.rand(n_batch, n_time, n_freq, n_sources)
    }

    target = {
        'source_magnitudes_a': 3 * torch.rand(n_batch, n_time, n_freq, n_sources),
        'source_magnitudes_b': .1 * torch.rand(n_batch, n_time, n_freq, n_sources)
    }

    weights = nn.ParameterList([
        nn.Parameter(torch.zeros(1)) for i in range(2)
    ])
    optimizer = optim.Adam(weights.parameters(), lr=1e-1)

    loss_dictionary = {
        'BigLoss': {
            'class': 'L1Loss',
            'weight': weights[0],
            'keys': {
                'estimates_a': 'input',
                'source_magnitudes_a': 'target',
            }
        },
        'SmallLoss': {
            'class': 'L1Loss',
            'weight': weights[1],
            'keys': {
                'estimates_b': 'input',
                'source_magnitudes_b': 'target',
            }
        }
    }

    closure = ml.train.closures.Closure(
        loss_dictionary, combination_approach='combine_by_multitask')
    loss_a = closure.compute_loss(output, target)

    for i in range(1000):
        optimizer.zero_grad()
        loss = closure.compute_loss(output, target)
        loss['loss'].backward()
        optimizer.step()

    
    var = []
    for p in weights.parameters():
        var.append(np.exp(-p.item()) ** 1)
    assert (
        var[0] * loss['BigLoss'] -
        var[1] * loss['SmallLoss']
    ) < 1e-2
    
    

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
        'mix_magnitude': torch.rand(n_batch, n_time, n_freq, 1),
        'ideal_binary_mask': assignments,
        'weights': weights,
        'source_magnitudes': torch.rand(n_batch, n_time, n_freq, 1, n_sources)
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

    # since closure expects an
    # engine within an ignite object, make a fake one
    engine = ml.train.create_train_and_validation_engines(closure)[0]
    init_loss = closure(engine, data)

    loss = None
    for i in range(100):
        loss = closure(engine, data)

    last_loss = loss

    for key in last_loss:
        assert last_loss[key] < init_loss[key]

    closure = ml.train.closures.ValidationClosure(
        loss_dictionary, model)

    for i in range(1):
        loss = closure(engine, data)

    for key in loss:
        assert np.allclose(loss[key], last_loss[key], 1e-1)
