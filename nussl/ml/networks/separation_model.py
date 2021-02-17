import os
import json
import inspect

import torch
from torch import nn
import numpy as np

from . import modules
from ... import __version__
import copy

def _remove_cache_from_tfms(transforms):
    """Helper function to remove cache from transforms.
    """
    from ... import datasets
    transforms = copy.deepcopy(transforms)

    if isinstance(transforms, datasets.transforms.Compose):
        for t in transforms.transforms:
            if isinstance(t, datasets.transforms.Cache):
                transforms.transforms.remove(t)

    return transforms


def _prep_metadata(metadata):
    """Helper function for preparing metadata before saving a model.
    """
    metadata = copy.deepcopy(metadata)
    if 'transforms' in metadata:
        metadata['transforms'] = _remove_cache_from_tfms(metadata['transforms'])
    return metadata

class SeparationModel(nn.Module):
    """
    SeparationModel takes a configuration file or dictionary that describes the model
    structure, which is some combination of MelProjection, Embedding, RecurrentStack,
    ConvolutionalStack, and other modules found in ``nussl.ml.networks.modules``. 

    References:
        Hershey, J. R., Chen, Z., Le Roux, J., & Watanabe, S. (2016, March).
        Deep clustering: Discriminative embeddings for segmentation and separation.
        In Acoustics, Speech and Signal Processing (ICASSP),
        2016 IEEE International Conference on (pp. 31-35). IEEE.

        Luo, Y., Chen, Z., Hershey, J. R., Le Roux, J., & Mesgarani, N. (2017, March).
        Deep clustering and conventional networks for music separation: Stronger together.
        In Acoustics, Speech and Signal Processing (ICASSP),
        2017 IEEE International Conference on (pp. 61-65). IEEE.

    Args:
        config: (str, dict) Either a config dictionary that defines the model and its
          connections, or the path to a json file containing the dictionary. If the
          latter, the path will be loaded and used.

    Attributes:
        config: (dict) The loaded config dictionary passed in upon init.
        connections: (list) A list of strings that define the connections as given
            in `config`.
        output: (list)

    See also:
        ml.register_module to register your custom modules with SeparationModel.

    Examples:
        >>> import nussl
        >>> config = nussl.ml.networks.builders.build_recurrent_dpcl(
        >>>     num_features=512, hidden_size=300, num_layers=3, bidirectional=True,
        >>>     dropout=0.3, embedding_size=20, 
        >>>     embedding_activation=['sigmoid', 'unit_norm'])
        >>>
        >>> model = SeparationModel(config)
    """
    def __init__(self, config, verbose=False):
        super(SeparationModel, self).__init__()
        if type(config) is str:
            if os.path.exists(config):
                with open(config, 'r') as f:
                    config = json.load(f)
            else:
                config = json.loads(config)

        self._validate_config(config)

        module_dict = {}
        self.input = {}
        for module_key in config['modules']:
            module = config['modules'][module_key]

            if 'class' in module:
                if module['class'] in dir(modules): 
                    class_func = getattr(modules, module['class'])
                    try:
                        module_snapshot = inspect.getsource(class_func)
                    except TypeError: # pragma: no cover
                        module_snapshot = (
                            "No module snapshot could be found. Did you define "
                            "your class in an interactive Python environment? "
                            "See https://bugs.python.org/issue12920 for more details."
                        )
                else:
                    class_func = getattr(nn, module['class'])
                    module_snapshot = f'pytorch v{torch.__version__} builtin'
                config['modules'][module_key]['module_snapshot'] = module_snapshot

                if 'args' not in module:
                    module['args'] = {}
                module_dict[module_key] = class_func(**module['args'])
            else:
                self.input[module_key] = module_key

        self.layers = nn.ModuleDict(module_dict)
        self.connections = config['connections']
        self.output_keys = config['output']
        self.config = config
        self.verbose = verbose
        self.metadata = {
            'config': config,
            'nussl_version': __version__
        }

    @staticmethod
    def _validate_config(config):
        expected_keys = ['connections', 'modules', 'name', 'output']
        got_keys = sorted(list(config.keys()))

        if got_keys != expected_keys:
            raise ValueError(
                f"Expected keys {expected_keys}, got {got_keys}")

        if not isinstance(config['modules'], dict):
            raise ValueError("config['modules'] must be a dict!")

        if not isinstance(config['connections'], list):
            raise ValueError("config['connections'] must be a list!")

        if not isinstance(config['output'], list):
            raise ValueError("config['output'] must be a list!")

        if not isinstance(config['name'], str):
            raise ValueError("config['name'] must be a string!")

    def forward(self, data=None, **kwargs):
        """
        Args:
            data: (dict) a dictionary containing the input data for the model. 
            Should match the input_keys in self.input.

        Returns:

        """
        data = {} if data is None else data
        data.update(kwargs)

        if not all(name in list(data) for name in list(self.input)):
            raise ValueError(
                f'Not all keys present in data! Needs {", ".join(self.input)}')
        output = {}

        for connection in self.connections:
            layer = self.layers[connection[0]]
            input_data = []
            kwargs = {}

            if len(connection) == 2:
                for c in connection[1]:
                    if isinstance(c, dict):
                        for key, val in c.items():
                            if val in output:
                                kwargs[key] = output[val]
                            elif val in data:
                                kwargs[key] = data[val]
                            elif key in data:
                                kwargs[key] = data[key]
                            else:
                                kwargs[key] = val
                    else:
                        input_data.append(output[c] if c in output else data[c])
            
            _output = layer(*input_data, **kwargs)
            added_keys = []
            if isinstance(_output, dict):
                for k in _output:
                    _key = f'{connection[0]}:{k}'
                    output[_key] = _output[k]
                    added_keys.append(_key)
            elif isinstance(_output, tuple):
                for i, val in enumerate(_output):
                    _key = f'{connection[0]}:{i}'
                    output[_key] = val
                    added_keys.append(_key)
            else:
                _key = connection[0]
                output[_key] = _output
                added_keys.append(_key)
            
            if self.verbose:
                input_shapes = []
                for d in input_data:
                    if torch.is_tensor(d):
                        input_shapes.append(tuple(d.shape))
                input_desc = ", ".join(map(str, input_shapes))
                output_desc = ", ".join(
                    [f"'{k}': {tuple(output[k].shape)}" for k in added_keys])
                stats = {}
                for k in added_keys:
                    stats[k] = {
                        'min': output[k].detach().min().item(),
                        'max': output[k].detach().max().item(),
                        'mean': output[k].detach().mean().item(),
                        'std': output[k].detach().std().item(),
                    }
                
                stats_desc = "\tStatistics:"
                for o in stats:
                    stats_desc += f"\n\t\t{o}"
                    for k in stats[o]: 
                        stats_desc += f"\n\t\t\t{k}: {stats[o][k]:.4f}"

                print(
                    f"{connection[1]} -> {connection[0]} \n"
                    f"\tTook inputs: {input_desc} \n"
                    f"\tProduced {output_desc} \n"
                    f"{stats_desc}"
                )
                
        return {o: output[o] for o in self.output_keys}

    @staticmethod
    def load(location):
        # Circular import
        from ...core.migration import SafeModelLoader

        safe_loader = SafeModelLoader()
        model_dict = safe_loader.load(location, 'cpu')
        metadata = model_dict['metadata']

        model = SeparationModel(metadata['config'])
        model.load_state_dict(model_dict['state_dict'])
        return model, metadata

    def save(self, location, metadata=None, train_data=None, 
             val_data=None, trainer=None):
        """
        Saves a SeparationModel into a location into a dictionary with the
        weights and model configuration.
        Args:
            location: (str) Where you want the model saved, as a path.
            metadata: (dict) Additional metadata to save along with the model. By default,
                model config and nussl version is saved as metadata.
            train_data: (BaseDataset) Dataset used for training. Metadata will be extracted
                from this object if it is passed into the save function, and saved 
                alongside the model.
            val_data: (BaseDataset) Dataset used for validation. Metadata will be extracted
                from this object if it is passed into the save function, and saved 
                alongside the model.
            trainer: (ignite.Engine) Engine used for training. Metadata will be extracted
                from this object if it is passed into the save function, and saved alongside
                the model.

        Returns:
            (str): where the model was saved.

        """
        save_dict = {
            'state_dict': self.state_dict(),
            'config': json.dumps(self.config)
        }

        metadata = metadata if metadata else {}
        metadata.update(self.metadata)

        if train_data is not None:
            dataset_metadata = {
                'stft_params': train_data.stft_params,
                'sample_rate': train_data.sample_rate,
                'num_channels': train_data.num_channels,
                'train_dataset': _prep_metadata(train_data.metadata),
            }
            metadata.update(dataset_metadata)

        try:
            metadata['val_dataset'] = _prep_metadata(val_data.metadata)
        except: # pragma: no cover
            pass
        
        if trainer is not None:
            train_metadata = {
                'trainer.state_dict': {
                    'epoch': trainer.state.epoch,
                    'epoch_length': trainer.state.epoch_length,
                    'max_epochs': trainer.state.max_epochs,
                    'output': trainer.state.output,
                    'metrics': trainer.state.metrics,
                    'seed': trainer.state.seed,
                },
                'trainer.state.epoch_history': trainer.state.epoch_history,
            }
            metadata.update(train_metadata)

        save_dict = {**save_dict, 'metadata': metadata}
        torch.save(save_dict, location)
        return location
    
    def __repr__(self):
        output = super().__repr__()
        num_parameters = 0
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += np.cumprod(p.size())[-1]
        output += '\nNumber of parameters: %d' % num_parameters
        return output
