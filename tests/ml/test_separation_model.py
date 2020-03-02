import nussl
import torch
from torch import nn
from nussl.ml.networks import SeparationModel, modules
import pytest
import json
import tempfile
import copy

n_features = 257

mi_config = {
    'modules': {
        'mix_magnitude': {},
        'log_spectrogram': {
            'class': 'AmplitudeToDB'
        },
        'normalization': {
            'class': 'BatchNorm'
        },
        'recurrent_stack': {
            'class': 'RecurrentStack',
            'args': {
                'num_features': n_features,
                'hidden_size': 50,
                'num_layers': 2,
                'bidirectional': True,
                'dropout': 0.3
            }
        },
        'mask': {
            'class': 'Embedding',
            'args': {
                'num_features': n_features,
                'hidden_size': 50 * 2,
                'embedding_size': 2,
                'activation': ['softmax']
            }
        },
        'estimates': {
            'class': 'Mask',
        },
    },
    'connections': [
        ('log_spectrogram', ('mix_magnitude',)),
        ('normalization', ('log_spectrogram',)),
        ('recurrent_stack', ('normalization',)),
        ('mask', ('recurrent_stack',)),
        ('estimates', ('mask', 'mix_magnitude'))
    ],
    'output': ['estimates']
}

dpcl_config = {
    'modules': {
        'mix_magnitude': {},
        'log_spectrogram': {
            'class': 'AmplitudeToDB'
        },
        'normalization': {
            'class': 'BatchNorm'
        },
        'recurrent_stack': {
            'class': 'RecurrentStack',
            'args': {
                'num_features': n_features,
                'hidden_size': 50,
                'num_layers': 2,
                'bidirectional': True,
                'dropout': 0.3
            }
        },
        'embedding': {
            'class': 'Embedding',
            'args': {
                'num_features': n_features,
                'hidden_size': 50 * 2,
                'embedding_size': 20,
                'activation': ['sigmoid', 'unit_norm']
            }
        },
    },
    'connections': [
        ('log_spectrogram', ('mix_magnitude',)),
        ('normalization', ('log_spectrogram',)),
        ('recurrent_stack', ('normalization',)),
        ('embedding', ('recurrent_stack',)),
    ],
    'output': ['embedding']
}

chimera_config = {
    'modules': {
        'mix_magnitude': {},
        'log_spectrogram': {
            'class': 'AmplitudeToDB'
        },
        'normalization': {
            'class': 'BatchNorm'
        },
        'recurrent_stack': {
            'class': 'RecurrentStack',
            'args': {
                'num_features': n_features,
                'hidden_size': 50,
                'num_layers': 2,
                'bidirectional': True,
                'dropout': 0.3
            }
        },
        'embedding': {
            'class': 'Embedding',
            'args': {
                'num_features': n_features,
                'hidden_size': 50 * 2,
                'embedding_size': 20,
                'activation': ['sigmoid', 'unit_norm']
            }
        },
        'mask': {
            'class': 'Embedding',
            'args': {
                'num_features': n_features,
                'hidden_size': 50 * 2,
                'embedding_size': 2,
                'activation': ['softmax']
            }
        },
        'estimates': {
            'class': 'Mask',
        },
    },
    'connections': [
        ('log_spectrogram', ('mix_magnitude',)),
        ('normalization', ('log_spectrogram',)),
        ('recurrent_stack', ('normalization',)),
        ('embedding', ('recurrent_stack',)),
        ('mask', ('recurrent_stack',)),
        ('estimates', ('mask', 'mix_magnitude',))
    ],
    'output': ['embedding', 'estimates']
}

gmm_unfold_config = {
    'modules': {
        'mix_magnitude': {},
        'log_spectrogram': {
            'class': 'AmplitudeToDB'
        },
        'normalization': {
            'class': 'BatchNorm'
        },
        'recurrent_stack': {
            'class': 'RecurrentStack',
            'args': {
                'num_features': n_features,
                'hidden_size': 50,
                'num_layers': 2,
                'bidirectional': True,
                'dropout': 0.3
            }
        },
        'embedding': {
            'class': 'Embedding',
            'args': {
                'num_features': n_features,
                'hidden_size': 50 * 2,
                'embedding_size': 20,
                'activation': ['sigmoid', 'unit_norm']
            }
        },
        'mask': {
            'class': 'GaussianMixture',
            'args': {
                'n_components': 2,
            }
        },
        'estimates': {
            'class': 'Mask',
        },
    },
    'connections': [
        ('log_spectrogram', ('mix_magnitude',)),
        ('normalization', ('log_spectrogram',)),
        ('recurrent_stack', ('normalization',)),
        ('embedding', ('recurrent_stack',)),
        ('mask', ('embedding',)),
        ('estimates', ('mask:resp', 'mix_magnitude',))
    ],
    'output': ['embedding', 'estimates']
}

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, data, **kwargs):
        return data

def test_separation_model_init():
    bad_config = {'not': {'the right keys'}}
    pytest.raises(ValueError, SeparationModel, bad_config)

    bad_config = {
        'modules': ['should be a dict'], 
        'connections': [],
        'output': []
    }
    pytest.raises(ValueError, SeparationModel, bad_config)

    bad_config = {
        'modules': mi_config['modules'],
        'connections': {'should be a list'},
        'output': []
    }
    pytest.raises(ValueError, SeparationModel, bad_config)

    bad_config = {
        'modules': mi_config['modules'],
        'connections': mi_config['connections'],
        'output': {'should be a list'}
    }
    pytest.raises(ValueError, SeparationModel, bad_config)

def test_separation_model_mask_inference(one_item):
    n_features = one_item['mix_magnitude'].shape[2]

    with tempfile.NamedTemporaryFile(suffix='.json', delete=True) as tmp:
        with open(tmp.name, 'w') as f:
            json.dump(mi_config, f)
        configs = [mi_config, tmp.name, json.dumps(mi_config)]

        for config in configs:    
            model = SeparationModel(config)

            bad_item = copy.deepcopy(one_item)
            bad_item.pop('mix_magnitude')
            pytest.raises(ValueError, model, bad_item)

            output = model(one_item)

            assert (
                output['estimates'].shape == (
                    one_item['mix_magnitude'].shape[:-1] + (2,))
            )
            assert  (
                torch.allclose(
                    output['estimates'].sum(dim=-1, keepdims=True), 
                    one_item['mix_magnitude']))

def test_separation_model_dpcl(one_item):
    n_features = one_item['mix_magnitude'].shape[2]

    # dpcl network
    with tempfile.NamedTemporaryFile(suffix='.json', delete=True) as tmp:
        with open(tmp.name, 'w') as f:
            json.dump(dpcl_config, f)
        configs = [dpcl_config, tmp.name, json.dumps(dpcl_config)]

        for config in configs:    
            model = SeparationModel(config)
            output = model(one_item)

            assert (
                output['embedding'].shape == (
                    one_item['mix_magnitude'].shape[:-1] + (20,)))

def test_separation_model_chimera(one_item):
    n_features = one_item['mix_magnitude'].shape[2]

    with tempfile.NamedTemporaryFile(suffix='.json', delete=True) as tmp:
        with open(tmp.name, 'w') as f:
            json.dump(chimera_config, f)
        configs = [chimera_config, tmp.name, json.dumps(chimera_config)]

        for config in configs:    
            model = SeparationModel(config)
            output = model(one_item)

            assert (
                output['estimates'].shape == (
                    one_item['mix_magnitude'].shape[:-1] + (2,))
            )
            assert  (
                torch.allclose(
                    output['estimates'].sum(dim=-1, keepdims=True), 
                    one_item['mix_magnitude']))

            assert (
                output['embedding'].shape == (
                    one_item['mix_magnitude'].shape[:-1] + (20,)))

def test_separation_model_gmm_unfold(one_item):
    n_features = one_item['mix_magnitude'].shape[2]

    with tempfile.NamedTemporaryFile(suffix='.json', delete=True) as tmp:
        with open(tmp.name, 'w') as f:
            json.dump(gmm_unfold_config, f)
        configs = [gmm_unfold_config, tmp.name, json.dumps(gmm_unfold_config)]

        for config in configs:    
            model = SeparationModel(config)
            output = model(one_item)

            assert (
                output['estimates'].shape == (
                    one_item['mix_magnitude'].shape[:-1] + (2,))
            )
            assert  (
                torch.allclose(
                    output['estimates'].sum(dim=-1, keepdims=True), 
                    one_item['mix_magnitude']))

            assert (
                output['embedding'].shape == (
                    one_item['mix_magnitude'].shape[:-1] + (20,)))

def test_separation_model_extra_modules(one_item):
    with tempfile.NamedTemporaryFile(suffix='.json', delete=True) as tmp:
        dpcl_config['modules']['test'] = {
            'class': 'MyModule'
        }
        dpcl_config['connections'].append(
            ('test', ('mix_magnitude', {'embedding': 'embedding'}))
        )
        dpcl_config['output'].append('test')
        with open(tmp.name, 'w') as f:
            json.dump(dpcl_config, f)
        configs = [dpcl_config, tmp.name, json.dumps(dpcl_config)]

        for config in configs:    
            model = SeparationModel(config, extra_modules=[MyModule])
            output = model(one_item)

            assert (
                output['embedding'].shape == (
                    one_item['mix_magnitude'].shape[:-1] + (20,)))

            assert torch.allclose(
                one_item['mix_magnitude'], output['test']
            )

def test_separation_model_save():
    model = SeparationModel(dpcl_config)

    with tempfile.NamedTemporaryFile(suffix='.pth', delete=True) as tmp:
        loc = model.save(tmp.name)
        checkpoint = torch.load(loc)

        new_model = SeparationModel(checkpoint['config'])
        new_model.load_state_dict(checkpoint['state_dict'])

        new_model_params = {}
        old_model_params = {}

        for name, param in new_model.named_parameters():
            new_model_params[name] = param

        for name, param in model.named_parameters():
            old_model_params[name] = param
        
        for key in new_model_params:
            assert torch.allclose(
                new_model_params[key],
                old_model_params[key]
            )

def test_separation_model_repr():
    model = SeparationModel(dpcl_config)
    print(model)
            