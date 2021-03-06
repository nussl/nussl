import nussl
import torch
from torch import nn
from nussl.ml.networks import SeparationModel, modules, builders
from nussl import datasets
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

open_unmix_like_config = builders.build_open_unmix_like(
    n_features, 50, 2, True, .4, 2, 1, add_embedding=True,
    embedding_size=20, embedding_activation=['sigmoid', 'unit_norm'],
)

end_to_end_real_config = builders.build_recurrent_end_to_end(
    512, 512, 128, 'sqrt_hann', 50, 2, 
   True, 0.3, 2, 'softmax', num_audio_channels=1, 
   mask_complex=False, rnn_type='lstm', 
   mix_key='mix_audio')

dual_path_recurrent_config = builders.build_dual_path_recurrent_end_to_end(
    64, 16, 8, 60, 30, 50, 2, True, 25, 2, 'sigmoid', 
)

end_to_end_complex_config = builders.build_recurrent_end_to_end(
    512, 512, 128, 'sqrt_hann', 50, 2, 
   True, 0.3, 2, 'softmax', num_audio_channels=1, 
   mask_complex=True, rnn_type='lstm', 
   mix_key='mix_audio')


gmm_unfold_config = copy.deepcopy(dpcl_config)
gmm_unfold_config['modules']['mask'] = {
    'class': 'GaussianMixtureTorch',
    'args': {
        'n_components': 2
    }
}

gmm_unfold_config['modules']['estimates'] = {'class': 'Mask',}

gmm_unfold_config['connections'].extend([
    ['mask', ['embedding', {'means': 'init_means'}]],
    ['estimates', ['mask:resp', 'mix_magnitude',]]
])

gmm_unfold_config['output'].append('estimates')

add_torch_module_config = copy.deepcopy(dpcl_config)
add_torch_module_config['modules']['mask'] = {
    'class': 'Linear',
    'args': {
        'in_features': 20,
        'out_features': 2
    }
}
add_torch_module_config['connections'].extend(
    [['mask', ['embedding']]]
)
add_torch_module_config['output'].append('mask')

split_config = copy.deepcopy(mi_config)
split_config['modules']['split'] = {
    'class': 'Split',
    'args': {
        'split_sizes': (100, 157),
        'dim': 2
    }
}

split_config['connections'].extend([
    ['split', ['estimates',]],
])

split_config['output'].append('split:0')
split_config['output'].append('split:1')


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, data=None, flip=False, **kwargs):
        if flip:
            data = -data
        return data

def test_separation_model_init():
    bad_config = {'not': {'the right keys'}}
    pytest.raises(ValueError, SeparationModel, bad_config)

    bad_config = {
        'name': 'BadModel',
        'modules': ['should be a dict'], 
        'connections': [],
        'output': []
    }
    pytest.raises(ValueError, SeparationModel, bad_config)

    bad_config = {
        'name': 'BadModel',
        'modules': mi_config['modules'],
        'connections': {'should be a list'},
        'output': []
    }
    pytest.raises(ValueError, SeparationModel, bad_config)

    no_name = {
        'modules': mi_config['modules'],
        'connections': mi_config['connections'],
        'output': []
    }
    pytest.raises(ValueError, SeparationModel, no_name)

    bad_name = {
        'name': 12345,
        'modules': mi_config['modules'],
        'connections': mi_config['connections'],
        'output': []
    }
    pytest.raises(ValueError, SeparationModel, bad_name)

    bad_config = {
        'name': 'BadModel',
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

def test_separation_end_to_end(one_item):
    for c in [end_to_end_real_config, end_to_end_complex_config]:
        with tempfile.NamedTemporaryFile(suffix='.json', delete=True) as tmp:
            with open(tmp.name, 'w') as f:
                json.dump(c, f)
            configs = [
                c, 
                tmp.name, 
                json.dumps(c)
            ]

            for config in configs:    
                model = SeparationModel(config)
                output = model(one_item)

                assert (
                    output['audio'].shape == one_item['source_audio'].shape
                )

def test_separation_dprnn(one_item):
    # dprnn network
    with tempfile.NamedTemporaryFile(suffix='.json', delete=True) as tmp:
        with open(tmp.name, 'w') as f:
            json.dump(dual_path_recurrent_config, f)
        configs = [
            dual_path_recurrent_config, 
            tmp.name, 
            json.dumps(dual_path_recurrent_config)
        ]

        for config in configs:    
            model = SeparationModel(config, verbose=True)
            output = model(one_item)

            assert (
                output['audio'].shape == one_item['source_audio'].shape
                )

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



def test_separation_model_open_unmix_like(one_item):
    n_features = one_item['mix_magnitude'].shape[2]

    with tempfile.NamedTemporaryFile(suffix='.json', delete=True) as tmp:
        with open(tmp.name, 'w') as f:
            json.dump(open_unmix_like_config, f)
        configs = [
            open_unmix_like_config, 
            tmp.name, 
            json.dumps(open_unmix_like_config)
        ]

        for config in configs:    
            model = SeparationModel(config)
            output = model(one_item)

            assert (
                output['estimates'].shape == (
                    one_item['mix_magnitude'].shape + (2,))
            )
            assert (
                output['mask'].shape == (
                    one_item['mix_magnitude'].shape + (2,))
            )
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
            one_item['init_means'] = torch.randn(
                one_item['mix_magnitude'].shape[0], 2, 20
            ).to(one_item['mix_magnitude'].device)
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

def test_separation_model_split(one_item):
    n_features = one_item['mix_magnitude'].shape[2]

    with tempfile.NamedTemporaryFile(suffix='.json', delete=True) as tmp:
        with open(tmp.name, 'w') as f:
            json.dump(split_config, f)
        configs = [split_config, tmp.name, json.dumps(split_config)]

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
                output['split:0'].shape[2] == 100)
            assert (
                output['split:1'].shape[2] == 157)


def test_separation_model_add_torch(one_item):
    n_features = one_item['mix_magnitude'].shape[2]
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=True) as tmp:
        with open(tmp.name, 'w') as f:
            json.dump(add_torch_module_config, f)

        configs = [
            add_torch_module_config, 
            tmp.name, 
            json.dumps(add_torch_module_config)
        ]

        for config in configs:    
            model = SeparationModel(config)
            output = model(one_item)

            assert (
                output['mask'].shape == (
                    one_item['mix_magnitude'].shape + (2,))
            )

            assert (
                output['embedding'].shape == (
                    one_item['mix_magnitude'].shape + (20,)))

def test_separation_model_extra_modules(one_item):
    with tempfile.NamedTemporaryFile(suffix='.json', delete=True) as tmp:
        dpcl_config['modules']['test'] = {
            'class': 'MyModule'
        }
        dpcl_config['connections'].append(
            ('test', ('mix_magnitude', {
                'embedding': 'embedding', 
                'flip': False
            }))
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

            model = SeparationModel(config)
            copy_one_item = copy.deepcopy(one_item)
            output = model(copy_one_item, flip=True)

            assert torch.allclose(
                one_item['mix_magnitude'], -output['test']
            )

def test_separation_model_save_and_load():
    model = SeparationModel(dpcl_config)

    tfms = datasets.transforms.Compose([
        datasets.transforms.PhaseSensitiveSpectrumApproximation(),
        datasets.transforms.ToSeparationModel(),
        datasets.transforms.Cache('tests/local/sep_model/cache')
    ])

    class DummyData:
        def __init__(self):
            self.stft_params = None
            self.sample_rate = None
            self.num_channels = None
            self.metadata = {'transforms': tfms}

    class DummyState:
        def __init__(self):
            self.epoch = 0
            self.epoch_length = 100
            self.max_epochs = 100
            self.output = None
            self.metrics = {}
            self.seed = None
            self.epoch_history = {}
    
    class DummyTrainer:
        def __init__(self):
            self.state = DummyState()

    dummy_data = DummyData()
    dummy_trainer = DummyTrainer()

    with tempfile.NamedTemporaryFile(suffix='.pth', delete=True) as tmp:

        loc = model.save(tmp.name, train_data=dummy_data, 
            val_data=dummy_data, trainer=dummy_trainer)
        new_model, metadata = SeparationModel.load(tmp.name)

        assert metadata['nussl_version'] == nussl.__version__

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

def test_separation_model_expose():
    class Model(nn.Module):
        def __init__(self, x):
            super().__init__()
            self.x = x

        def forward(self, y):
            return self.x + y

    nussl.ml.register_module(Model)

    config = {
        'modules': {
            'model': {
                'class': 'Model',
                'args': {
                    'x': 10
                },
                'expose_forward': True
            }
        },
        'connections': [],
        'output': [],
        'name': 'Model',
    }

    separation_model = nussl.ml.SeparationModel(config)
    assert separation_model(y=5) == 15



def test_separation_model_repr_and_verbose(one_item):
    model = SeparationModel(end_to_end_real_config, verbose=True)
    print(model)
    model(one_item)
