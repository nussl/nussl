import pytest 
from nussl.separation import primitive, SeparationException
import numpy as np
import os
import nussl

REGRESSION_PATH = 'tests/separation/regression/primitive/'
os.makedirs(REGRESSION_PATH, exist_ok=True)

def test_timbre_clustering(
    drum_and_vocals, 
    check_against_regression_data
):
    np.random.seed(0)
    drum, vocals = drum_and_vocals
    drum.resample(16000)
    vocals.resample(16000)

    mix = drum + vocals

    separator = primitive.TimbreClustering(
        mix, 2, 100, mask_type='binary')
    estimates = separator()

    evaluator = nussl.evaluation.BSSEvalScale(
        [drum, vocals], estimates, 
        source_labels=['drums', 'vocals'], compute_permutation=True
    )
    scores = evaluator.evaluate()

    reg_path = os.path.join(REGRESSION_PATH, 'timbre_clustering.json')
    check_against_regression_data(scores, reg_path)


def test_ft2d(
    music_mix_and_sources,
    check_against_regression_data
):
    mix, sources = music_mix_and_sources
    vox = sources['vocals']
    acc = sources['group0']

    pytest.raises(SeparationException, primitive.FT2D, mix, 
        filter_approach='none of the above')

    config = [
        ({}, 'defaults'),
        ({'quadrants_to_keep': (1, 3)}, 'quadrants_13'),
        ({'quadrants_to_keep': (2, 4)}, 'quadrants_24'),
        ({'filter_approach': 'original'}, 'original'),
        ({'use_bg_2dft': False}, 'foreground'),
        ({'mask_type': 'binary'}, 'binary')
    ]

    for kwargs, name in config:
        sep = primitive.FT2D(mix, **kwargs)
        estimates = sep()

        evaluator = nussl.evaluation.BSSEvalScale(
            [acc, vox], estimates, 
            source_labels=['acc', 'vocals'], 
        )

        scores = evaluator.evaluate()

        reg_path = os.path.join(
            REGRESSION_PATH, f'ft2d_{name}.json')
        check_against_regression_data(scores, reg_path)

def test_hpss(
    drum_and_vocals, 
    check_against_regression_data
):
    drum, vocals = drum_and_vocals
    mix = drum + vocals

    for mask_type in ['soft', 'binary']:
        hpss = primitive.HPSS(mix, mask_type=mask_type)
        estimates = hpss()

        evaluator = nussl.evaluation.BSSEvalScale(
            [vocals, drum], estimates, 
            source_labels=['vocals', 'drum'], 
        )

        scores = evaluator.evaluate()

        reg_path = os.path.join(
            REGRESSION_PATH, f'hpss_{mask_type}.json')
        check_against_regression_data(scores, reg_path)
