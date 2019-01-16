from enum import Enum
import torch
from torch.utils.data import sampler
from torch import nn
from .. import datasets
from . import loss

class Datasets(Enum):
    MIXSOURCEFOLDER = datasets.MixSourceFolder
    SCAPER = datasets.Scaper

class Samplers(Enum):
    SEQUENTIAL = sampler.SequentialSampler
    RANDOM = sampler.RandomSampler

class LossFunctions(Enum):
    DPCL = loss.DeepClusteringLoss
    PIT = loss.PermutationInvariantLoss
    L1 = nn.L1Loss
    MSE = nn.MSELoss
    KL = nn.KLDivLoss

class Optimizers(Enum):
    ADAM = torch.optim.Adam
    RMSPROP = torch.optim.RMSprop
    SGD = torch.optim.SGD
