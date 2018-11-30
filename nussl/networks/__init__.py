from .. import torch_imported, ImportErrorClass
if torch_imported:
    from .separation_model import SeparationModel
else:
    class SeparationModel(ImportErrorClass):
        def __init__(self):
            super(SeparationModel, self).__init__('pytorch')