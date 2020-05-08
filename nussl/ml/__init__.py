"""
Machine Learning
================

SeparationModel
---------------

.. autoclass:: nussl.ml.SeparationModel
    :members:
    :autosummary:

Building blocks for SeparationModel
-----------------------------------

.. automodule:: nussl.ml.modules
    :members:
    :autosummary:

.. automodule:: nussl.ml.cluster
    :members:
    :autosummary:

Helpers for common deep networks
--------------------------------

.. automodule:: nussl.ml.networks.builders
    :members:
    :autosummary:

Confidence measures
-------------------

.. automodule:: nussl.ml.confidence
    :members:
    :autosummary:

Training
--------

.. automodule:: nussl.ml.train
    :members:
    :autosummary:
"""

from .networks import SeparationModel
from .networks import modules

from . import train
from . import unfold
from . import cluster
from . import confidence

from sklearn.decomposition import NMF

def register_module(module):
    """
    Your custom modules can be registered with nussl's SeparationModel
    via this function. For example, if you have some module `ExampleModule`,
    you can register it as follows:

    .. code-block:: python

        class ExampleModule(nn.Module):
        def forward(self, data):
            data = data * 2
            return data

        ml.register_module(ExampleModule)

    You can now use the name `ExampleModule` in the config for a SeparationModel.
    
    Args:
        module ([type]): [description]
    """
    setattr(modules, module.__name__, module)


def register_loss(custom_loss):
    """
    Your custom loss functions can be registered with nussl's Closure
    via this function. For example, if you have some loss `ExampleLoss`,
    you can register it as follows:

    .. code-block:: python

        class ExampleLoss(nn.Module):
            DEFAULT_KEYS = {'key1': 'arg1', 'key2': 'arg2'}
            def forward(self, arg1, arg2):
                # do something
                return 0

        ml.register_loss(ExampleLoss)

    You can now use the name `ExampleLoss` in a loss dictionary that is passed
    to a closure.

    
    Args:
        custom_loss ([type]): [description]
    """
    setattr(train.loss, custom_loss.__name__, custom_loss)
