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
