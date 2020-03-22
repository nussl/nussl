import pytest 
from nussl.separation import factorization, SeparationException
import numpy as np
import os
import nussl
import copy

REGRESSION_PATH = 'tests/separation/regression/factorization/'
os.makedirs(REGRESSION_PATH, exist_ok=True)


def test_rpca(
    music_mix_and_sources, 
    check_against_regression_data
):
    nussl.utils.seed(0)
    mix, sources = music_mix_and_sources
    mix = copy.deepcopy(mix)
    vox = copy.deepcopy(sources['vocals'])
    acc = copy.deepcopy(sources['group0'])

    config = [
        ({}, 'defaults'),
        ({'epsilon': 10}, 'high_epsilon'),
        ({'mask_type': 'binary'}, 'binary'),
    ]

    for kwargs, name in config:
        rpca = factorization.RPCA(mix, **kwargs)
        estimates = rpca()
        
        evaluator = nussl.evaluation.BSSEvalScale(
            [acc, vox], estimates, 
            source_labels=['acc', 'vocals'], 
        )

        scores = evaluator.evaluate()

        reg_path = os.path.join(
            REGRESSION_PATH, f'rpca_{name}.json')
        check_against_regression_data(scores, reg_path)