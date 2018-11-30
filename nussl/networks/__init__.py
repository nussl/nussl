from .. import torch_imported, ImportErrorClass
if torch_imported:
    from .separation_model import SeparationModel
else:
    class SeparationModel(ImportErrorClass):
        def __init__(self, *args, **kwargs):
            super(DeepSeparation, self).__init__('pytorch')