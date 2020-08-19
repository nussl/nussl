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

    Or if you're using permutation invariant loss but need to specify arguments to the
    loss function being wrapped by PIT, you can do this:

    .. code-block:: python

        loss_dictionary = {
            'PITLoss': {
                'class': 'PermutationInvariantLoss',
                'keys': {'audio': 'estimates', 'source_audio': 'targets'},
                'args': [{
                    'class': 'SISDRLoss',
                    'kwargs': {'scaling': False}
                }]
            }
        }


    If you have your own loss function classes you wish to use, you can pass those
    into the loss dictionary and make them discoverable by the closure by using
    `ml.register_loss.`

    Args:
        loss_dictionary (dict): Dictionary of losses described above.
        combination_approach (str): How to combine losses, if there are multiple
         losses. The default is that the losses will be combined via a weighted
         sum ('combine_by_sum'). Can also do 'combine_by_multiply'. Defaults to
         'combine_by_sum'.
        args: Positional arguments to ``combination_approach``.
        kwargs: Keyword arguments to ``combination_approach``.

    See also:
        ml.register_loss to register your loss functions with this closure.
    """

    def __init__(self, loss_dictionary, combination_approach='combine_by_sum', 
                 *args, **kwargs):
        loss_dictionary = self._validate_loss_dictionary(loss_dictionary)

        self.combination_func = getattr(self, combination_approach)
        self.args = args
        self.kwargs = kwargs

        self.losses = []
        for key, val in loss_dictionary.items():
            _loss_name = val['class'] if 'class' in val else key
            loss_class = getattr(loss, _loss_name)
            weight = 1 if 'weight' not in val else val['weight']
            keys = loss_class.DEFAULT_KEYS if 'keys' not in val else val['keys']
            args = [] if 'args' not in val else copy.deepcopy(val['args'])
            kwargs = {} if 'kwargs' not in val else copy.deepcopy(val['kwargs'])
            if _loss_name in ['CombinationInvariantLoss', 'PermutationInvariantLoss']:
                if isinstance(args[0], str):
                    args[0] = getattr(loss, args[0])()
                elif isinstance(args[0], dict):
                    arg_class = getattr(loss, args[0]['class'])
                    args_to_loss = [] if 'args' not in args[0] else args[0]['args']
                    kwargs_to_loss = {} if 'kwargs' not in args[0] else args[0]['kwargs']
                    args[0] = arg_class(*args_to_loss, **kwargs_to_loss)

            _loss = (loss_class(*args, **kwargs), weight, keys, key)
            self.losses.append(_loss)

    @staticmethod
    def _validate_loss_dictionary(loss_dictionary):
        if not isinstance(loss_dictionary, dict):
            raise ClosureException(
                "loss_dictionary must be a dictionary specifying the "
                "class and arguments for each loss function! ")

        for key, val in loss_dictionary.items():
            _loss = val['class'] if 'class' in val else key
            if _loss not in dir(loss):
                raise ClosureException(
                    f"Loss function {_loss} not found in loss which has {dir(loss)}")

            if not isinstance(val, dict):
                raise ClosureException(
                    "Each key in loss dictionary must point to a dict!")

            for val_key in val:
                if val_key not in ['weight', 'keys', 'args', 'kwargs', 'class']:
                    raise ClosureException(
                        f"{key} in loss_dictionary not in ['weight', 'args', 'kwargs'")

                elif val_key == 'weight':
                    if not isinstance(val[val_key], (float, int)) and not torch.is_tensor(val[val_key]):
                        raise ClosureException(f"weight can only be an int or a float")

                elif val_key == 'args':
                    if not isinstance(val[val_key], list):
                        raise ClosureException(f"args must be a list")

                elif val_key == 'kwargs':
                    if not isinstance(val[val_key], dict):
                        raise ClosureException("kwargs must be a dict")
        
        return loss_dictionary

    def __call__(self, engine, data):
        raise NotImplementedError()

    def combine_by_multitask(self, loss_output):
        """
        Implements a multitask learning objective [1] where each loss
        is weighted by a learned parameter with the following
        function:

        combined_loss = \sum_i exp(-weight_i) * loss_i + weight_i

        where i indexes each loss. The weights come from the loss 
        dictionary and can point to nn.Parameter teensors that get 
        learned jointly with the model.

        References:

        [1] Kendall, Alex, Yarin Gal, and Roberto Cipolla. 
            "Multi-task learning using uncertainty to weigh losses 
            for scene geometry and semantics." Proceedings of the 
            IEEE conference on computer vision and pattern recognition. 2018.
        """
        combined_loss = 0
        for _, weight, _, name in self.losses:
            sigma = torch.exp(-weight)
            combined_loss += sigma * loss_output[name] + weight
        return combined_loss

    def combine_by_multiply(self, loss_output):
        combined_loss = 1
        for _, weight, _, name in self.losses:
            combined_loss *= weight * loss_output[name]
        return combined_loss

    def combine_by_sum(self, loss_output):
        combined_loss = 0
        for _, weight, _, name in self.losses:
            # if the weight is 0, then the loss is just for
            # monitoring and we won't bother summing with it, 
            # in case its shape doesnt match.
            if weight != 0:
                combined_loss += weight * loss_output[name]
        return combined_loss

    def compute_loss(self, output, target):
        loss_output = {}
        output.update(target)

        for loss_obj, weight, keys, name in self.losses:
            kwargs = {keys[k]: output[k] for k in keys}
            loss_output[name] = loss_obj(**kwargs)
        
        loss_output['loss'] = self.combination_func(
            loss_output, *self.args, **self.kwargs)
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

    def __init__(self, loss_dictionary, optimizer, model, *args, **kwargs):
        super().__init__(loss_dictionary, *args, **kwargs)
        self.optimizer = optimizer
        self.model = model

        # Save about training metadata to model.info
        self.model.metadata['optimizer'] = {
            'name': type(optimizer).__name__,
            'params': optimizer.defaults  # All of the settings are stored here.
        }
        self.model.metadata['loss_dictionary'] = loss_dictionary

    def _fire_event(self, engine, output, event):
        if engine is not None:
            if engine.state is not None:
                engine.state.model_output = output
            engine.fire_event(event)

    def __call__(self, engine, data):
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(data)

        loss_ = self.compute_loss(output, data)
        loss_['loss'].backward()
        self._fire_event(engine, output, BackwardsEvents.BACKWARDS_COMPLETED)
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

    def __init__(self, loss_dictionary, model, *args, **kwargs):
        super().__init__(loss_dictionary, *args, **kwargs)
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
