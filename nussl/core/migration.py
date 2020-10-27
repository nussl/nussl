import torch
import json

from .. import __version__, STFTParams
from ..separation.base import SeparationException
from ..datasets import transforms as tfm
from ..evaluation import BSSEvalV4, BSSEvalScale


class SafeModelLoader(object):
    """
    Loads a nussl model and populates the metadata with defaults if
    """

    # Expected topology of the model's metadata as of nussl version 1.1.3
    _v1_1_3_metadata = {
        'config': {
            'connections': list,
            'modules': dict,
            'name': str,
            'output': list
        },
        'evaluation': None,
        'loss_dictionary': dict,
        'num_channels': int,
        'nussl_version': str,
        'optimizer': {
            'name': str,
            'params': dict
        },
        'sample_rate': int,
        'stft_params': STFTParams,
        'train_dataset': {
            'folder': str,
            'name': str,
            'num_channels': int,
            'sample_rate': int,
            'stft_params': STFTParams,
            'transforms': tfm.Compose
        },
        'trainer.state.epoch_history': dict,
        'trainer.state_dict': {
            'epoch': int,
            'epoch_length': int,
            'max_epochs': int,
            'metrics': dict,
            'output': dict,
            'seed': None
        },
        'val_dataset': {
            'folder': str,
            'name': str,
            'num_channels': int,
            'sample_rate': int,
            'stft_params': STFTParams,
            'transforms': tfm.Compose
        },
    }

    expected_metadata = _v1_1_3_metadata

    def __init__(self):
        """
        Drop in replacement for torch.load(). Will load a model and return
        a model_dict that is populated
        Args:
            model_path (str): Path to a nussl-saved model.
            device (str):
            expected_eval (str): Either 'BSSEvalScale' or 'BSSEvalV4'. Will
                look for & populate missing eval keys with the format of either
                of those eval methods.

        Returns:
            model_dict (dict):

        """
        self.current_version = __version__
        self.eval = None

    def load(self, model_path, device='cpu', expected_eval='BssEvalScale'):
        model_dict = torch.load(model_path, map_location=device)
        metadata = model_dict['metadata']
        metadata = self._get_moved(metadata, model_dict, model_path)
        self.eval = expected_eval
        metadata = self._validate_and_populate(metadata)
        metadata['config'] = json.dumps(metadata['config'])
        model_dict['metadata'] = metadata
        return model_dict

    @staticmethod
    def _get_moved(metadata, model_dict, model_path):
        model_dict_version = model_dict.get('nussl_version', None)
        metadata_version = metadata.get('nussl_version', None)

        if metadata_version is not None:
            saved_version = metadata_version
        else:
            saved_version = model_dict_version

        if saved_version is None:
            raise SeparationException(f"Failed loading model. Expected to find "
                                      f"'nussl_version' in {model_path}.")

        metadata['nussl_version'] = saved_version

        if 'config' in model_dict:
            metadata['config'] = json.loads(model_dict['config'])

        if 'transforms' in metadata:
            if 'train_dataset' not in metadata:
                metadata['train_dataset'] = {}
            metadata['train_dataset']['transforms'] = metadata['transforms']

        return metadata

    def _load_eval(self, eval_dict):
        """Helper function to load eval dictionary safely."""
        if self.eval.lower() == 'bssevalv4':
            keys = BSSEvalV4.keys
        else:
            keys = BSSEvalScale.keys

        stats_keys = ['mean', 'median', 'std']

        result = {}
        for k in keys:
            if k not in eval_dict:
                stats = {s: 'UNAVAILABLE' for s in stats_keys}
            else:
                stats = {}
                for s in stats_keys:
                    if s in eval_dict[k]:
                        stats[s] = eval_dict[k][s]
                    else:
                        stats[s] = 'UNAVAILABLE'
            result[k] = stats
        return result

    @staticmethod
    def _load_types(expected_type, key, val):
        """Safe load for values where the value is a type in self.expected_metadata"""
        if val is not None:
            if type(val) != expected_type:
                raise SeparationException(f'Expected type {expected_type} '
                                          f'for key {key} but got {type(val)}!')
            return val
        else:
            return 'UNAVAILABLE'

    def _validate_and_populate(self, received):
        """Safe load for metadata according to the expected metadata."""
        result = {}
        for key, expected_val in self.expected_metadata.items():
            val = received.get(key, None)

            if key == 'evaluation':
                eval_dict = received.get('evaluation', {})
                result['evaluation'] = self._load_eval(eval_dict)
            elif type(expected_val) == type:
                result[key] = self._load_types(expected_val, key, val)

            elif type(expected_val) == dict:
                next_dict = {} if val is None else val
                sub_result = {}
                for sub_key, type_ in expected_val.items():
                    sub_val = next_dict.get(sub_key, None)
                    sub_result[sub_key] = self._load_types(type_, sub_key, sub_val)
                result[key] = sub_result

        return result