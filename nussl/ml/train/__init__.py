"""
Training
--------

.. autofunction:: nussl.ml.train.create_train_and_validation_engines

.. autofunction:: nussl.ml.train.add_tensorboard_handler

.. autofunction:: nussl.ml.train.cache_dataset

.. autofunction:: nussl.ml.train.add_validate_and_checkpoint

.. autofunction:: nussl.ml.train.add_stdout_handler

.. autoclass:: nussl.ml.train.ValidationEvents
    :members:

.. autoclass:: nussl.ml.train.BackwardsEvents
    :members:

Loss functions
--------------

.. automodule:: nussl.ml.train.loss
    :members:
    :autosummary:

Closures
--------

.. automodule:: nussl.ml.train.closures
    :members:
    :autosummary:

"""

from .trainer import (
    create_train_and_validation_engines,
    add_tensorboard_handler,
    cache_dataset,
    add_validate_and_checkpoint,
    add_stdout_handler,
    ValidationEvents,
    BackwardsEvents
)

from . import loss
from . import closures
