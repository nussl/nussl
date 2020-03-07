import torch
from . import loss

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

    Args:
        loss_dictionary (dict): Dictionary of losses described above.

    """
    def __init__(self, loss_dictionary):
        if not isinstance(loss_dictionary, dict):
            raise ClosureException(
                "loss_dictionary must be a dictionary specifying the "
                "class and arguments for each loss function! ")
        
        self.losses = []
        for key, val in loss_dictionary.items():
            loss_class = getattr(loss, key)
            weight = 1 if 'weight' not in val else val['weight']
            keys = loss_class.DEFAULT_KEYS  if 'keys' not in val else val['keys']
            args = [] if 'args' not in val else val['args']
            kwargs = {} if 'kwargs' not in val else val['kwargs']
            if key in ['CombinationInvariantLoss', 'PermutationInvariantLoss']:
                args[0] = getattr(loss, args[0])()

            _loss = (
                loss_class(*args, **kwargs), weight, keys, key)
            self.losses.append(_loss)
    
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

        loss = self.compute_loss(output, data)

        loss['loss'].backward()
        self.optimizer.step()

        loss = {key: loss[key].item() for key in loss}

        return loss

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
            loss = self.compute_loss(output, data)
            loss = {key: loss[key].item() for key in loss}
        return loss

class ClosureException(Exception):
    """ 
    Exception class for errors when working with closures in nussl.
    """
    pass
