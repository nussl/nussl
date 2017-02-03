#!/usr/bin/env python
# -*- coding: utf-8 -*-

import audio_signal
import spectral_utils
import constants

import copy
import json
import utils
import numpy as np
import inspect


class SeparationBase(object):
    """Base class for all separation algorithms in nussl.

    Do not call this. It will not do anything.

    Parameters:
        input_audio_signal (:obj:`AudioSignal`). ``AudioSignal`` object.
                            This will always make a copy of the provided AudioSignal object.
    """

    def __init__(self, input_audio_signal):
        self._audio_signal = None

        if input_audio_signal is not None:
            self.audio_signal = input_audio_signal
        else:
            self.audio_signal = audio_signal.AudioSignal()

    @property
    def sample_rate(self):
        """(int): Sample rate of ``self.audio_signal``.
        Literally ``self.audio_signal.sample_rate``.
        """
        return self.audio_signal.sample_rate

    @property
    def stft_params(self):
        """(:obj:`StftParams`): ``StftParams`` of ``self.audio_signal``
        Literally ``self.audio_signal.stft_params``.
        """
        return self.audio_signal.stft_params

    @property
    def audio_signal(self):
        """(:obj:`AudioSignal`): Copy of the ``AudioSignal`` object passed in upon initialization.
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
        """Makes ``AudioSignal`` objects after separation algorithm is run

        Raises:
            NotImplementedError: Cannot call base class
        """
        raise NotImplementedError('Cannot call base class.')

    def to_json(self):
        return json.dumps(self, default=SeparationBase._to_json_helper)

    @staticmethod
    def _to_json_helper(o):
        if not isinstance(o, SeparationBase):
            raise TypeError

        d = copy.copy(o.__dict__)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = utils.json_ready_numpy_array(v)
            if isinstance(v, audio_signal.AudioSignal) or isinstance(v, spectral_utils.StftParams):
                d[k] = v.to_json()

        d['__class__'] = o.__class__.__name__
        d['__module__'] = o.__module__
        if 'self' in d:
            del d['self']

        return d

    @classmethod
    def from_json(cls, json_string):
        sep_decoder = SeparationBaseDecoder(cls)
        return sep_decoder.decode(json_string)

    def __call__(self):
        self.run()

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
    """ Object to decode a ``SeparationBase``-derived object from JSON serialization.
    You should never have to instantiate this object by hand.
    """

    def __init__(self, separation_class):
        self.separation_class = separation_class
        json.JSONDecoder.__init__(self, object_hook=self.json_separation_decoder)

    def json_separation_decoder(self, json_dict):
        if '__class__' in json_dict:
            class_name = json_dict.pop('__class__')
            module_name = json_dict.pop('__module__')
            if class_name != self.separation_class.__name__ or module_name != self.separation_class.__module__:
                raise TypeError

            # load the module and import the class
            module = __import__(module_name)
            class_ = getattr(module, class_name)

            # we know 'input_audio_signal' is always the first argument
            signal_json = json_dict.pop('_audio_signal')  # this is the AudioSignal object
            signal = audio_signal.AudioSignal.from_json(signal_json)

            # get the rest of the required arguments
            signature = inspect.getargspec(class_.__init__)
            # first arg is covered above (2), and we don't want the non-defaults (-len(signature.defaults))
            required_args = signature.args[2:-len(signature.defaults)]
            args = dict((k.encode('ascii'), json_dict[k]) for k in required_args)

            # make a new instance of separation class
            seperator = class_(signal, **args)

            # fill out the rest of the fields
            for k, v in json_dict.items():
                if isinstance(v, dict) and constants.NUMPY_JSON_KEY in v:
                    seperator.__dict__[k] = utils.json_numpy_obj_hook(v[constants.NUMPY_JSON_KEY])
                elif isinstance(v, basestring) and audio_signal.__name__ in v: # TODO: python3-ify this
                    seperator.__dict__[k] = audio_signal.AudioSignal.from_json(v)
                else:
                    seperator.__dict__[k] = v if not isinstance(v, unicode) else v.encode('ascii')

            return seperator
        else:
            return json_dict
