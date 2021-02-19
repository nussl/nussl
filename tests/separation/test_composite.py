from nussl.separation.base.separation_base import SeparationBase
import pytest
from nussl.separation import (
    primitive,
    factorization,
    composite,
    SeparationException
)
import numpy as np
import os
import nussl
import copy

REGRESSION_PATH = 'tests/separation/regression/composite/'
os.makedirs(REGRESSION_PATH, exist_ok=True)


def test_ensemble_clustering(
        music_mix_and_sources,
        check_against_regression_data
):
    mix, sources = music_mix_and_sources
    mix = copy.deepcopy(mix)
    vox = copy.deepcopy(sources['vocals'])
    acc = copy.deepcopy(sources['drums+bass+other'])

    separators = [
        primitive.FT2D(mix),
        factorization.RPCA(mix),
        primitive.HPSS(mix),
    ]

    weights = [3, 3, 1]
    returns = [[1], [1], [0]]

    fixed_centers = np.array([
        [0 for i in range(sum(weights))],
        [1 for i in range(sum(weights))],
    ])

    config = [
        ({}, 'defaults'),
        ({
             'init': fixed_centers,
             'fit_clusterer': False,
             'weights': weights,
             'returns': returns
         }, 'fixed_means'),
        ({
             'extracted_feature': 'estimates',
             'weights': weights,
             'returns': returns
         }, 'use_estimates'),
    ]

    for kwargs, name in config:
        nussl.utils.seed(0)
        ensemble = composite.EnsembleClustering(
            mix, 2, separators, **kwargs)
        estimates = ensemble()

        evaluator = nussl.evaluation.BSSEvalScale(
            [acc, vox], estimates,
            source_labels=['acc', 'vocals'],
            compute_permutation=True
        )

        scores = evaluator.evaluate()

        reg_path = os.path.join(
            REGRESSION_PATH, f'ensemble_clustering_{name}.json')
        check_against_regression_data(scores, reg_path, atol=1e-2)

    pytest.raises(SeparationException, composite.EnsembleClustering,
                  mix, 2, separators, extracted_feature='none of the above')

    pytest.raises(SeparationException, composite.EnsembleClustering,
                  mix, 2, separators, weights=[1, 1])
    pytest.raises(SeparationException, composite.EnsembleClustering,
                  mix, 2, separators, returns=[[1], [1]])


def test_overlap_add():
    def random_noise(duration, ch, kind):
        if kind == 'ones':
            x = np.ones((int(ch), int(duration * 44100)))
        elif kind == 'random':
            x = np.random.randn(int(ch), int(duration * 44100))
        signal = nussl.AudioSignal(
            audio_data_array=x, 
            sample_rate=44100
        )
        signal.peak_normalize()
        return signal

    # Check the static methods
    mix = random_noise(10, 2, 'random')
    windows = composite.OverlapAdd.collect_windows(mix, 2, 1)
    recombined = composite.OverlapAdd.overlap_and_add(windows, mix, 2, 1)

    assert np.allclose(recombined.audio_data, mix.audio_data)

    class DoNothing(SeparationBase):
        def __init__(self, input_audio_signal):
            super().__init__(input_audio_signal)
        def run(self):
            return 
        def make_audio_signals(self):
            sig = self.audio_signal.make_copy_with_audio_data(self.audio_signal.audio_data)
            return [sig]
    
    mix = random_noise(1, 2, 'random')
    do_nothing = DoNothing(mix)
    overlap_add = composite.OverlapAdd(do_nothing)
    estimates = overlap_add()

    assert np.allclose(estimates[0].audio_data, mix.audio_data) 

    for k in ['ones', 'random']:
        for dur in [1.5, 10, 30, 95, 101]:
            for ch in range(1, 3):
                mix = random_noise(dur, ch, k)

                before_mix = copy.deepcopy(mix)
                do_nothing = DoNothing(mix)
                overlap_add = composite.OverlapAdd(do_nothing, window_length=1)
                estimates = overlap_add()

                assert before_mix == mix
                assert np.allclose(estimates[0].audio_data, mix.audio_data)
