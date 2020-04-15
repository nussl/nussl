import os
import json

import torch
from torch import nn
import numpy as np

from . import modules
from ... import __version__

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

    See also:
        ml.register_module to register your custom modules with SeparationModel.

    Examples:
        >>> config = nussl.ml.networks.builders.build_recurrent_dpcl(
        >>>     num_features=512, hidden_size=300, num_layers=3, bidirectional=True,
        >>>     dropout=0.3, embedding_size=20, 
        >>>     embedding_activation=['sigmoid', 'unit_norm'])
        >>>
        >>> model = SeparationModel(config)
    """
    def __init__(self, config):
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
                else:
                    class_func = getattr(nn, module['class'])
                if 'args' not in module:
                    module['args'] = {}
                module_dict[module_key] = class_func(**module['args'])
            else:
                self.input[module_key] = module_key

        self.layers = nn.ModuleDict(module_dict)
        self.connections = config['connections']
        self.output_keys = config['output']
        self.config = config

    @staticmethod
    def _validate_config(config):
        expected_keys = ['connections', 'modules', 'output']
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

    def forward(self, data):
        """
        Args:
            data: (dict) a dictionary containing the input data for the model. 
            Should match the input_keys in self.input.

        Returns:

        """
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
                            else:
                                kwargs[key] = val
                    else:
                        input_data.append(output[c] if c in output else data[c])
            _output = layer(*input_data, **kwargs)
            if isinstance(_output, dict):
                for k in _output:
                    output[f'{connection[0]}:{k}'] = _output[k]
            elif isinstance(_output, tuple):
                for i, val in enumerate(_output):
                    output[f'{connection[0]}:{i}'] = val
            else:
                output[connection[0]] = _output
                
        return {o: output[o] for o in self.output_keys}

    def save(self, location, metadata=None):
        """
        Saves a SeparationModel into a location into a dictionary with the
        weights and model configuration.
        Args:
            location: (str) Where you want the model saved, as a path.

        Returns:
            (str): where the model was saved.

        """
        save_dict = {
            'state_dict': self.state_dict(),
            'config': json.dumps(self.config)
        }

        metadata = metadata if metadata else {}
        metadata['nussl_version'] = __version__
        save_dict = {**save_dict, **metadata}
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
