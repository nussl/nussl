#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .. import torch_imported, vamp_imported, ImportErrorClass

# Import base classes
from .masks import *
from .separation_base import SeparationBase
from .mask_separation_base import MaskSeparationBase

# Median based algorithms
from .repet import Repet
from .repet_sim import RepetSim
from .ft2d import FT2D
from .hpss import HPSS

median_algorithms = [Repet, RepetSim, FT2D, HPSS]

# Melody-based methods
if vamp_imported:
    from .melodia import Melodia
else:
    class Melodia(ImportErrorClass):
        def __init__(self, *args, **kwargs):
            super(Melodia, self).__init__('vamp')

melody_algorithms = [Melodia]

# Spatialization based methods
from .duet import Duet
from .projet import Projet
from .multichannel_wiener_filter import MultichannelWienerFilter

spatialization_algorithms = [Duet, Projet]

# Benchmark algorithms
from .ideal_mask import IdealMask
from .high_low_pass_filter import HighLowPassFilter

benchmark_algorithms = [IdealMask, HighLowPassFilter]

# Composite algorithms
from .overlap_add import OverlapAdd
composite_instruments = [OverlapAdd]

# Matrix factorization and component analysis
from .nmf_mfcc import NMF_MFCC
from .ica import ICA
from .rpca import RPCA

nmf_algorithms = [NMF_MFCC]
component_analysis_algorithms = [ICA, RPCA]

# Clustering algorithms
from .clustering import SpatialClustering, PrimitiveClustering

# Deep learning algorithms
if torch_imported:
    from .deep_mask_estimation import DeepMaskEstimation
    from .clustering import DeepClustering
else:
    class DeepMaskEstimation(ImportErrorClass):
        def __init__(self, *args, **kwargs):
            super().__init__('pytorch')
    class DeepClustering(ImportErrorClass):
        def __init__(self, *args, **kwargs):
            super().__init__('pytorch')

deep_learning_algorithms = [DeepClustering, DeepMaskEstimation]

all_separation_algorithms = [median_algorithms, melody_algorithms, spatialization_algorithms,
                             benchmark_algorithms, nmf_algorithms, component_analysis_algorithms,
                             deep_learning_algorithms]

all_separation_algorithms = [val for sublist in all_separation_algorithms for val in sublist]
