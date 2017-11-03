#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import inspect
import json
import warnings

import numpy as np

from ..core import utils
from ..core import audio_signal
from ..core import constants
from ..core.audio_signal import AudioSignal


class SeparationBase(object):
    """Base class for all separation algorithms in nussl.

    Do not call this. It will not do anything.

    Parameters:
        input_audio_signal (:class:`audio_signal.AudioSignal`). :class:`audio_signal.AudioSignal` object.
                            This will always make a copy of the provided AudioSignal object.
    """

    def __init__(self, input_audio_signal):
        if not isinstance(input_audio_signal, AudioSignal):
            raise ValueError('input_audio_signal is not an AudioSignal object!')

        self._audio_signal = None

        if input_audio_signal is not None:
            self.audio_signal = input_audio_signal
        else:
            self.audio_signal = AudioSignal()

        if not self.audio_signal.has_data:
            warnings.warn('input_audio_signal has no data!')

            # initialize to empty arrays so that we don't crash randomly
            self.audio_signal.audio_data = np.array([])
            self.audio_signal.stft_data = np.array([[]])

    @property
    def sample_rate(self):
        """(int): Sample rate of :attr:`audio_signal`.
        Literally :attr:`audio_signal.sample_rate`.
        """
        return self.audio_signal.sample_rate

    @property
    def stft_params(self):
        """(:class:`spectral_utils.StftParams`): :class:`spectral_utils.StftParams` of :attr:`audio_signal`
        Literally :attr:`audio_signal.stft_params`.
        """
        return self.audio_signal.stft_params

    @property
    def audio_signal(self):
        """(:class:`audio_signal.AudioSignal`): Copy of the :class:`audio_signal.AudioSignal` object passed in 
        upon initialization.
        """
        return self._audio_signal

    @audio_signal.setter
    def audio_signal(self, input_audio_signal):
        self._audio_signal = copy.copy(input_audio_signal)

    def plot(self, output_name, **kwargs):
        """Plots relevant data for separation algorithm

        Raises:
            NotImplementedError: Cannot call base class
        """
        raise NotImplementedError('Cannot call base class.')

    def run(self):
        """Runs separation algorithm

        Raises:
            NotImplementedError: Cannot call base class
        """
        raise NotImplementedError('Cannot call base class.')

    def make_audio_signals(self):
        """Makes :class:`audio_signal.AudioSignal` objects after separation algorithm is run

        Raises:
            NotImplementedError: Cannot call base class
        """
        raise NotImplementedError('Cannot call base class.')

    def to_json(self):
        """
        Outputs JSON from the data stored in this object.
        
        Returns:
            (str) a JSON string containing all of the information to restore this object exactly as it was when this
            was called.
            
        See Also:
            :func:`from_json` to restore a JSON frozen object.

        """
        return json.dumps(self, default=SeparationBase._to_json_helper)

    def __str__(self):
        return self.__class__.__name__

    @staticmethod
    def _to_json_helper(o):
        if not isinstance(o, SeparationBase):
            raise TypeError('SeparationBase._to_json_helper() got foreign object!')

        d = copy.copy(o.__dict__)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = utils.json_ready_numpy_array(v)
            elif hasattr(v, 'to_json'):
                d[k] = v.to_json()
            elif isinstance(v, (list, tuple, set)) and any(hasattr(itm, 'to_json') for itm in v):
                s = []
                for itm in v:
                    if hasattr(itm, 'to_json'):
                        s.append(itm.to_json())
                    else:
                        s.append(itm)
                d[k] = s

        d['__class__'] = o.__class__.__name__
        d['__module__'] = o.__module__
        if 'self' in d:
            del d['self']

        return d

    @classmethod
    def from_json(cls, json_string):
        """
        Creates a new :class:`SeparationBase` object from the parameters stored in this JSON string.
        
        Args:
            json_string (str): A JSON string containing all the data to create a new :class:`SeparationBase` 
                object.

        Returns:
            (:class:`SeparationBase`) A new :class:`SeparationBase` object from the JSON string.
            
        See Also:
            :func:`to_json` to make a JSON string to freeze this object.

        """
        sep_decoder = SeparationBaseDecoder(cls)
        return sep_decoder.decode(json_string)

    def __call__(self):
        return self.run()

    def __repr__(self):
        return self.__class__.__name__ + ' instance'

    def __eq__(self, other):
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                if not np.array_equal(v, other.__dict__[k]):
                    return False
            elif k == 'self':
                pass
            elif v != other.__dict__[k]:
                return False
        return True

    def __ne__(self, other):
        return not self == other


class SeparationBaseDecoder(json.JSONDecoder):
    """ Object to decode a :class:`SeparationBase`-derived object from JSON serialization.
    You should never have to instantiate this object by hand.
    """

    def __init__(self, separation_class):
        self.separation_class = separation_class
        json.JSONDecoder.__init__(self, object_hook=self._json_separation_decoder)

    def _inspect_json_and_create_new_instance(self, json_dict):
        class_name = json_dict.pop('__class__')
        module_name = json_dict.pop('__module__')
        if class_name != self.separation_class.__name__ or module_name != self.separation_class.__module__:
            raise TypeError('Expected {}.{} but got {}.{} from json!'.format(self.separation_class.__module__,
                                                                             self.separation_class.__name__,
                                                                             module_name, class_name))

        # load the module and import the class
        module = __import__(module_name)
        class_ = getattr(module, class_name)

        if '_audio_signal' not in json_dict:
            raise TypeError('JSON string from {} does not have an AudioSignal object!'.format(class_name))

        # we know 'input_audio_signal' is always the first argument
        signal_json = json_dict.pop('_audio_signal')  # this is the AudioSignal object
        signal = AudioSignal.from_json(signal_json)

        # get the rest of the required arguments
        signature = inspect.getargspec(class_.__init__)
        # first arg is covered above (2), and we don't want the non-defaults (-len(signature.defaults))
        non_required_args = 0 if signature.defaults is None else len(signature.defaults)
        required_args = signature.args[2:-non_required_args]
        args = dict((k.encode('ascii'), json_dict[k]) for k in required_args)

        # make a new instance of separation class
        separator = class_(signal, **args)

        return json_dict, separator

    def _json_separation_decoder(self, json_dict):
        """
        Helper method for :class:`SeparationBaseDecoder`. Don't you worry your pretty little head about this.

        NEVER CALL THIS DIRECTLY!!

        Args:
            json_dict (dict): JSON dictionary provided by `object_hook`

        Returns:
            A new :class:`SeparationBase`-derived object from JSON serialization

        """
        if '__class__' in json_dict and '__module__' in json_dict:
            json_dict, separator = self._inspect_json_and_create_new_instance(json_dict)

            # fill out the rest of the fields
            for k, v in json_dict.items():
                if isinstance(v, dict) and constants.NUMPY_JSON_KEY in v:
                    separator.__dict__[k] = utils.json_numpy_obj_hook(v[constants.NUMPY_JSON_KEY])
                elif isinstance(v, (str, bytes)) and audio_signal.__name__ in v:  # TODO: test this
                    separator.__dict__[k] = AudioSignal.from_json(v)
                else:
                    separator.__dict__[k] = v if not isinstance(v, unicode) else v.encode('ascii')

            return separator
        else:
            return json_dict
