import pytest 
from nussl.separation import primitive, SeparationException
import numpy as np
import os
import nussl
import copy
from importlib import reload

REGRESSION_PATH = 'tests/separation/regression/primitive/'
os.makedirs(REGRESSION_PATH, exist_ok=True)

def test_timbre_clustering(
    drum_and_vocals, 
    check_against_regression_data
):
    np.random.seed(0)
    drum, vocals = drum_and_vocals
    drum = copy.deepcopy(drum)
    vocals = copy.deepcopy(vocals)

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
    acc = sources['drums+bass+other']

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

def test_repet(
    music_mix_and_sources, 
    check_against_regression_data
):
    mix, sources = music_mix_and_sources
    vox = sources['vocals']
    acc = sources['drums+bass+other']

    pytest.raises(
        SeparationException, primitive.Repet, mix, min_period=.8, 
            max_period=8, period=3)

    pytest.raises(SeparationException, 
        primitive.Repet.find_repeating_period_simple,
        np.random.rand(100), 101, 5)

    config = [
        ({}, 'defaults'),
        ({'mask_type': 'binary'}, 'binary'),
        ({'period': 3}, 'period_set'),
    ]

    for kwargs, name in config:
        repet = primitive.Repet(mix, **kwargs)
        estimates = repet()

        evaluator = nussl.evaluation.BSSEvalScale(
            [acc, vox], estimates, 
            source_labels=['acc', 'vocals'], 
        )

        scores = evaluator.evaluate()

        reg_path = os.path.join(
            REGRESSION_PATH, f'repet_{name}.json')
        check_against_regression_data(scores, reg_path)

def test_melodia(
    drum_and_vocals, 
    check_against_regression_data,
):
    drum, vocals = drum_and_vocals
    mix = drum + vocals
    
    nussl.vamp_imported = False
    pytest.raises(
        SeparationException, primitive.Melodia, mix)
    
    nussl.vamp_imported = True

    os.environ['VAMP_PATH'] = (
        f"{os.path.abspath('tests/vamp/melodia_osx')}:"
        f"{os.path.abspath('tests/vamp/melodia_linux')}"
    )

    melodia = primitive.Melodia(mix, mask_type='soft')

    config = [
        ({'mask_type': 'binary'}, 'binary'),
        ({'mask_type': 'binary', 'apply_vowel_filter': True}, 'vf_binary'),
         ({'mask_type': 'binary', 'add_lower_octave': True}, 'octave_binary'),
        ({'mask_type': 'soft'}, 'soft'),
    ]

    for kwargs, name in config:
        melodia = primitive.Melodia(mix, **kwargs)
        estimates = melodia()

        evaluator = nussl.evaluation.BSSEvalScale(
            [drum, vocals], estimates, 
            source_labels=['drum', 'vocals'], 
        )

        scores = evaluator.evaluate()
        reg_path = os.path.join(
            REGRESSION_PATH, f'melodia_{name}.json')
        check_against_regression_data(scores, reg_path)

def test_repet_sim(
    music_mix_and_sources, 
    check_against_regression_data
):
    mix, sources = music_mix_and_sources
    mix = copy.deepcopy(mix)
    vox = copy.deepcopy(sources['vocals'])
    acc = copy.deepcopy(sources['drums+bass+other'])

    config = [
        ({}, 'defaults'),
        ({'mask_type': 'binary'}, 'binary'),
        ({'similarity_threshold': .5}, 'high_similarity_threshold')
    ]

    for kwargs, name in config:
        repet_sim = primitive.RepetSim(mix, **kwargs)
        estimates = repet_sim()
        
        evaluator = nussl.evaluation.BSSEvalScale(
            [acc, vox], estimates, 
            source_labels=['acc', 'vocals'], 
        )

        scores = evaluator.evaluate()

        reg_path = os.path.join(
            REGRESSION_PATH, f'repet_sim_{name}.json')
        check_against_regression_data(scores, reg_path)