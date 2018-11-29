from torch import nn
import json
from . import modules
import torch
import numpy as np
from itertools import chain

class SeparationModel(nn.Module):
    def __init__(self, config):
        """
        SeparationModel takes a configuration file or dictionary that describes the model
        structure, which is some combination of MelProjection, Embedding, RecurrentStack,
        ConvolutionalStack, and other modules found in networks.modules. The configuration file
        can be built using the helper functions in config.builders:
            - build_dpcl_config: Builds the original deep clustering network, mapping each
                time-frequency point to an embedding of some size. Takes as input a
                log_spectrogram.
            - build_mi_config: Builds a "traditional" mask inference network that maps the mixture
                spectrogram to source estimates.  Takes as input a log_spectrogram and a
                magnitude_spectrogram.
            - build_chimera_config: Builds a Chimera network with a mask inference head and a
                deep clustering head to map. A combination of MI and DPCL. Takes as input a
                log_spectrogram and a magnitude_spectrogram.

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
            config: (str, dict) Either a config dictionary built from one of the helper functions,
                or the path to a json file containing a config built from the helper functions.

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
            with open(config, 'r') as f:
                config = json.load(f)
        module_dict = {}
        self.input = {}
        for module_key in config['modules']:
            module = config['modules'][module_key]
            if 'input_shape' not in module:
                class_func = getattr(modules, module['class'])
                module_dict[module_key] = class_func(**module['args'])
            else:
                self.input[module_key] = module['input_shape']

        self.layers = nn.ModuleDict(module_dict)
        self.connections = config['connections']
        self.output_keys = config['output']

    def forward(self, data):
        """
        Args:
            data: (dict) a dictionary containing the input data for the model. Should match the input_keys
                in self.input.

        Returns:

        """
        if not all(name in list(data) for name in list(self.input)):
            raise ValueError("Not all keys present in data! Needs {}".format(', '.join(self.input)))
        output = {}
        all_connections = [[connection[0]] + connection[1] for connection in self.connections]
        all_connections = list(chain.from_iterable(all_connections))
        marked_for_deletion = []

        for connection in self.connections:
            layer = self.layers[connection[0]]
            input_data = []
            for c in connection[1]:
                input_data.append(output[c] if c in output else data[c])
                all_connections.remove(c)
                if c not in all_connections:
                    marked_for_deletion.append(c)
            output[connection[0]] = layer(*input_data)

            for c in marked_for_deletion:
                if c in output:
                    output.pop(c)

        return {o: output[o] for o in self.output_keys}

    def project_assignments(self, data):
        if 'mel_projection' in self.layers:
            data = self.layers['mel_projection'](data)
            data = data.clamp(0.0, 1.0)
        return data

    def save(self, location):
        """
        Saves a SeparationModel into a location into a dictionary with the
        weights and model configuration.
        Args:
            location: (str) Where you want the model saved, as a path.

        Returns:

        """
        torch.save({'state_dict': self.state_dict(),
                    'model': self}, location)
    
    def __repr__(self):
        output = super().__repr__()
        num_parameters = 0
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += np.cumprod(p.size())[-1]
        output += '\nNumber of parameters: %d' % num_parameters
        return output