from .. import torch_imported, ImportErrorClass

if torch_imported:
    from .networks import SeparationModel
    from .networks import modules
    from . import train
    from . import datasets
    from . import config
else:
    class SeparationModel(ImportErrorClass):
        def __init__(self, *args, **kwargs):
            super(SeparationModel, self).__init__('pytorch')