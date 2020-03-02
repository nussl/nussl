from torch import nn
import json
from . import modules
import torch
import numpy as np
from itertools import chain

class SeparationModel(nn.Module):
    def __init__(self, config, extra_modules=None):
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

            extra_modules (list): A list of classes that are to be tacked onto the default
            classes that are used to instantiate each nn.Module that is used in the
            network.

        Examples:
            >>> args = {
            >>>    'num_frequencies': 512,
            >>>    'num_mels': 128,
            >>>    'sample_rate': 44100,
            >>>    'hidden_size': 300,
            >>>    'bidirectional': True,
            >>>    'num_layers': 4,
            >>>    'embedding_size': 20,
            >>>    'num_sources': 4,
            >>>    'embedding_activation': ['sigmoid', 'unitnorm'],
            >>>    'mask_activation': ['softmax']
            >>> }
            >>> config = helpers.build_chimera_config(args)
            >>> with open('config.json', 'w') as f:
            >>>    json.dump(config, f)
            >>> model = SeparationModel('config.json')
            >>> test_data = np.random.random((1, 100, 512))
            >>> data = torch.from_numpy(test_data).float()
            >>> output = model({'log_spectrogram': data,
            >>>                'magnitude_spectrogram': data})

        """
        super(SeparationModel, self).__init__()
        if type(config) is str:
            if 'json' in config:
                with open(config, 'r') as f:
                    config = json.load(f)
            else:
                config = json.loads(config)

        # Add extra modules to modules
        if extra_modules:
            for name in dir(extra_modules):
                if name not in dir(modules):
                    setattr(
                        modules, 
                        name,
                        getattr(extra_modules, name)
                    )

        module_dict = {}
        self.input = {}
        for module_key in config['modules']:
            module = config['modules'][module_key]
            if 'input_shape' not in module:
                class_func = getattr(modules, module['class'])
                if 'args' not in module:
                    module['args'] = {}
                module_dict[module_key] = class_func(**module['args'])
            else:
                self.input[module_key] = module['input_shape']

        self.layers = nn.ModuleDict(module_dict)
        self.connections = config['connections']
        self.output_keys = config['output']
        self.config = config

    def forward(self, data):
        """
        Args:
            data: (dict) a dictionary containing the input data for the model. Should match the input_keys
                in self.input.

        Returns:

        """
        if not all(name in list(data) for name in list(self.input)):
            raise ValueError(f'Not all keys present in data! Needs {", ".join(self.input)}')
        output = {}

        for connection in self.connections:
            layer = self.layers[connection[0]]
            input_data = []
            kwargs = {}

            if len(connection) == 2:
                for c in connection[1]:
                    if isinstance(c, dict):
                        for key, val in c.items():
                            kwargs[key] = output[val] if val in output else data[val]
                    else:
                        input_data.append(output[c] if c in output else data[c])
            _output = layer(*input_data, **kwargs)
            if isinstance(_output, dict):
                for k in _output:
                    output[f'{connection[0]}:{k}'] = _output[k]
            else:
                output[connection[0]] = _output
                
        return {o: output[o] for o in self.output_keys}

    def project_data(self, data, clamp=False):
        if 'mel_projection' in self.layers:
            data = self.layers['mel_projection'](data)
            if clamp:
                data = data.clamp(0.0, 1.0)
        return data

    def save(self, location, metadata=None):
        """
        Saves a SeparationModel into a location into a dictionary with the
        weights and model configuration.
        Args:
            location: (str) Where you want the model saved, as a path.

        Returns:

        """
        save_dict = {
            'state_dict': self.state_dict(),
            'config': json.dumps(self.config)
        }
        save_dict = {**save_dict, **(metadata if metadata else {})}
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
