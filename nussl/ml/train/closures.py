import copy

import torch

from . import loss
from .trainer import BackwardsEvents


class Closure(object):
    """
    Closures are used with ignite Engines to train a model given an optimizer 
    and a set of loss functions. Closures perform forward passes of models given 
    the input data. The loss is computed via ``self.compute_loss``. 
    The forward pass is implemented via the objects ``__call__`` function. 

    This closure object provides a way to define the loss functions you want to
    use to train your model as a loss dictionary that is structured as follows:

    .. code-block:: python

        loss_dictionary = {
            'LossClassName': {
                'weight': [how much to weight the loss in the sum, defaults to 1],
                'keys': [key mapping items in dictionary to arguments to loss],
                'args': [any positional arguments to the loss class],
                'kwargs': [keyword arguments to the loss class],
            }
        }

    The keys value will default to ``LossClassName.DEFAULT_KEYS``, which can be
    found in ``nussl.ml.train.loss`` within each available class. Here's an example
    of a Chimera loss combining deep clustering with permutation invariant L1 loss:

    .. code-block:: python 

        loss_dictionary = {
            'DeepClusteringLoss': {
                'weight': .2,
            },
            'PermutationInvariantLoss': {
                'weight': .8,
                'args': ['L1Loss']
            }
        }

    If you have your own loss function classes you wish to use, you can pass those
    into the loss dictionary and make them discoverable by the closure by using
    `ml.register_loss.`

    Args:
        loss_dictionary (dict): Dictionary of losses described above.

    See also:
        ml.register_loss to register your loss functions with this closure.

    """

    def __init__(self, loss_dictionary):
        loss_dictionary = self._validate_loss_dictionary(loss_dictionary)

        self.losses = []
        for key, val in loss_dictionary.items():
            loss_class = getattr(loss, key)
            weight = 1 if 'weight' not in val else val['weight']
            keys = loss_class.DEFAULT_KEYS if 'keys' not in val else val['keys']
            args = [] if 'args' not in val else val['args']
            kwargs = {} if 'kwargs' not in val else val['kwargs']
            if key in ['CombinationInvariantLoss', 'PermutationInvariantLoss']:
                args[0] = getattr(loss, args[0])()

            _loss = (loss_class(*args, **kwargs), weight, keys, key)
            self.losses.append(_loss)

    @staticmethod
    def _validate_loss_dictionary(loss_dictionary):
        if not isinstance(loss_dictionary, dict):
            raise ClosureException(
                "loss_dictionary must be a dictionary specifying the "
                "class and arguments for each loss function! ")

        for key, val in loss_dictionary.items():
            if key not in dir(loss):
                raise ClosureException(
                    f"Loss function {key} not found in loss which has {dir(loss)}")

            if not isinstance(val, dict):
                raise ClosureException(
                    "Each key in loss dictionary must point to a dict!")

            for val_key in val:
                if val_key not in ['weight', 'args', 'kwargs']:
                    raise ClosureException(
                        f"{key} in loss_dictionary not in ['weight', 'args', 'kwargs'")

                elif val_key == 'weight':
                    if not isinstance(val[val_key], float) and not isinstance(val[val_key], int):
                        raise ClosureException(f"weight can only be an int or a float")

                elif val_key == 'args':
                    if not isinstance(val[val_key], list):
                        raise ClosureException(f"args must be a list")

                elif val_key == 'kwargs':
                    if not isinstance(val[val_key], dict):
                        raise ClosureException("kwargs must be a dict")

        return copy.deepcopy(loss_dictionary)

    def __call__(self, engine, data):
        raise NotImplementedError()

    def compute_loss(self, output, target):
        loss_output = {}
        output.update(target)
        loss_output['loss'] = 0

        for loss_obj, weight, keys, name in self.losses:
            kwargs = {keys[k]: output[k] for k in keys}
            loss_output[name] = loss_obj(**kwargs)
            loss_output['loss'] += weight * loss_output[name]
        return loss_output


class TrainClosure(Closure):
    """
    This closure takes an optimization step on a SeparationModel object given a
    loss.
    
    Args:
        loss_dictionary (dict): Dictionary containing loss functions and specification.
        optimizer (torch Optimizer): Optimizer to use to train the model.
        model (SeparationModel): The model to be trained.
    """

    def __init__(self, loss_dictionary, optimizer, model):
        super().__init__(loss_dictionary)
        self.optimizer = optimizer
        self.model = model

    def __call__(self, engine, data):
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(data)

        loss_ = self.compute_loss(output, data)
        loss_['loss'].backward()
        engine.fire_event(BackwardsEvents.BACKWARDS_COMPLETED)
        self.optimizer.step()
        loss_ = {key: loss_[key].item() for key in loss_}

        return loss_


class ValidationClosure(Closure):
    """
    This closure validates the model on some data dictionary.
    
    Args:
        loss_dictionary (dict): Dictionary containing loss functions and specification.
        model (SeparationModel): The model to be validated.
    """

    def __init__(self, loss_dictionary, model):
        super().__init__(loss_dictionary)
        self.model = model

    def __call__(self, engine, data):
        with torch.no_grad():
            self.model.eval()
            output = self.model(data)
            loss_ = self.compute_loss(output, data)
            loss_ = {key: loss_[key].item() for key in loss_}
        return loss_


class ClosureException(Exception):
    """ 
    Exception class for errors when working with closures in nussl.
    """
    pass
