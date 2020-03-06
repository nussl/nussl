from enum import Enum
import torch
from torch.utils.data import sampler
from torch import nn
from . import loss

class Samplers(Enum):
    SEQUENTIAL = sampler.SequentialSampler
    RANDOM = sampler.RandomSampler

class LossFunctions(Enum):
    DPCL = loss.DeepClusteringLoss
    PIT = loss.PermutationInvariantLoss
    CIT = loss.CombinationInvariantLoss
    L1 = nn.L1Loss
    MSE = nn.MSELoss
    KL = nn.KLDivLoss

class Optimizers(Enum):
    ADAM = torch.optim.Adam
    RMSPROP = torch.optim.RMSprop
    SGD = torch.optim.SGD