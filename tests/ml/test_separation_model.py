import nussl
import torch
from torch import nn
from nussl.ml.networks import SeparationModel, modules, builders
import pytest
import json
import tempfile
import copy

n_features = 257

mi_config = builders.build_recurrent_mask_inference(
    n_features, 50, 2, True, 0.3, 2, 'softmax', 
)

dpcl_config = builders.build_recurrent_dpcl(
    n_features, 50, 2, False, 0.3, 20, ['sigmoid', 'unit_norm']
)

chimera_config = builders.build_recurrent_chimera(
    n_features, 50, 2, True, 0.3, 20, ['sigmoid', 'unit_norm'], 
    2, 'softmax', 
)

gmm_unfold_config = copy.deepcopy(dpcl_config)
gmm_unfold_config['modules']['mask'] = {
    'class': 'GaussianMixtureTorch',
    'args': {
        'n_components': 2
    }
}

gmm_unfold_config['modules']['estimates'] = {'class': 'Mask',}

gmm_unfold_config['connections'].extend(
    [['mask', ['embedding',]],
    ['estimates', ['mask:resp', 'mix_magnitude',]]]
)

gmm_unfold_config['output'].append('estimates')

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
                    one_item['mix_magnitude'].shape + (2,))
            )
            assert  (
                torch.allclose(
                    output['estimates'].sum(dim=-1), 
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
                    one_item['mix_magnitude'].shape + (20,)))

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
                    one_item['mix_magnitude'].shape + (2,))
            )
            assert  (
                torch.allclose(
                    output['estimates'].sum(dim=-1), 
                    one_item['mix_magnitude']))

            assert (
                output['embedding'].shape == (
                    one_item['mix_magnitude'].shape + (20,)))

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
                    one_item['mix_magnitude'].shape + (2,))
            )
            assert  (
                torch.allclose(
                    output['estimates'].sum(dim=-1), 
                    one_item['mix_magnitude']))

            assert (
                output['embedding'].shape == (
                    one_item['mix_magnitude'].shape + (20,)))

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

        nussl.ml.register_module(MyModule)

        for config in configs:    
            model = SeparationModel(config)
            output = model(one_item)

            assert (
                output['embedding'].shape == (
                    one_item['mix_magnitude'].shape + (20,)))

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
            