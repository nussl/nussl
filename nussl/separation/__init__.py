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
        def __init__(self, **kwargs):
            super(Melodia, self).__init__('vamp')

melody_algorithms = [Melodia]

# Spatialization based methods
from .duet import Duet
from .projet import Projet

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

# Deep learning algorithms
if torch_imported:
    from .deep_clustering import DeepClustering
else:
    class DeepClustering(ImportErrorClass):
        def __init__(self, **kwargs):
            super(DeepClustering, self).__init__('pytorch')

deep_learning_algorithms = [DeepClustering]

all_separation_algorithms = [median_algorithms, melody_algorithms, spatialization_algorithms,
                             benchmark_algorithms, nmf_algorithms, component_analysis_algorithms,
                             deep_learning_algorithms]

all_separation_algorithms = [val for sublist in all_separation_algorithms for val in sublist]

__all__ = ['SeparationBase', 'MaskSeparationBase',
           'all_separation_algorithms',
           'median_algorithms', 'Repet', 'RepetSim', 'HPSS', 'FT2D',
           'melody_algorithms', 'Melodia',
           'spatialization_algorithms', 'Duet', 'Projet',
           'benchmark_algorithms', 'IdealMask', 'HighLowPassFilter',
           'composite_instruments', 'OverlapAdd',
           'nmf_algorithms', 'NMF_MFCC',
           'component_analysis_algorithms', 'ICA', 'RPCA',
           'deep_learning_algorithms', 'DeepClustering']
