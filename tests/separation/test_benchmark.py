import nussl
from nussl.separation import SeparationException
import pytest
import os
import json

REGRESSION_PATH = 'tests/separation/regression/benchmark/'
os.makedirs(REGRESSION_PATH, exist_ok=True)


def test_high_low_pass(
        music_mix_and_sources,
        check_against_regression_data
):
    mix, sources = music_mix_and_sources
    sources = list(sources.values())

    hlp = nussl.separation.benchmark.HighLowPassFilter(
        mix, 100)
    estimates = hlp()

    evaluator = nussl.evaluation.BSSEvalScale(
        sources, estimates, compute_permutation=True)
    scores = evaluator.evaluate()

    reg_path = os.path.join(REGRESSION_PATH, 'high_low_pass.json')
    check_against_regression_data(scores, reg_path)


def test_ideal_binary_mask(
        music_mix_and_sources,
        check_against_regression_data
):
    mix, sources = music_mix_and_sources
    ibm = nussl.separation.benchmark.IdealBinaryMask(mix, sources)

    sources = list(sources.values())
    ibm = nussl.separation.benchmark.IdealBinaryMask(mix, sources)
    estimates = ibm()

    evaluator = nussl.evaluation.BSSEvalScale(
        sources, estimates, compute_permutation=True)
    scores = evaluator.evaluate()

    reg_path = os.path.join(REGRESSION_PATH, 'ideal_binary_mask.json')
    check_against_regression_data(scores, reg_path)

    pytest.raises(SeparationException, nussl.separation.benchmark.IdealBinaryMask,
                  mix, 'not a list or dict')


def test_ideal_ratio_mask(
        music_mix_and_sources,
        check_against_regression_data
):
    mix, sources = music_mix_and_sources
    irm = nussl.separation.benchmark.IdealRatioMask(mix, sources)
    sources = list(sources.values())

    pytest.raises(SeparationException, nussl.separation.benchmark.IdealRatioMask,
            mix, sources, approach='none of the above')

    for approach in ['psa', 'msa']:
        irm = nussl.separation.benchmark.IdealRatioMask(
            mix, sources, approach=approach)
        estimates = irm()

        evaluator = nussl.evaluation.BSSEvalScale(
            sources, estimates, compute_permutation=True)
        scores = evaluator.evaluate()

        reg_path = os.path.join(
            REGRESSION_PATH, f'ideal_ratio_mask_{approach}.json')
        check_against_regression_data(scores, reg_path)

        pytest.raises(SeparationException, nussl.separation.benchmark.IdealRatioMask,
                      mix, 'not a list or dict')


def test_wiener_filter(
        music_mix_and_sources,
        check_against_regression_data
):
    nussl.utils.seed(0)
    mix, sources = music_mix_and_sources
    irm = nussl.separation.benchmark.IdealRatioMask(mix, sources)
    sources = list(sources.values())

    # get initial estimates
    ibm = nussl.separation.benchmark.IdealBinaryMask(mix, sources)
    _estimates = ibm()

    configs = [
        ({'mask_type': 'binary'}, 'binary'),
        ({'mask_type': 'soft'}, 'soft')
    ]

    for kwargs, name in configs:
        # refine them with a wiener filter
        wf = nussl.separation.benchmark.WienerFilter(
            mix, _estimates, **kwargs)
        estimates = wf()

        evaluator = nussl.evaluation.BSSEvalScale(
            sources, estimates, compute_permutation=True)
        scores = evaluator.evaluate()

        reg_path = os.path.join(
            REGRESSION_PATH, f'wiener_ibm_{name}.json')
        check_against_regression_data(scores, reg_path)

    pytest.raises(SeparationException, nussl.separation.benchmark.WienerFilter,
                  mix, 'not a list or dict')


def test_mix_as_estimate(
        music_mix_and_sources,
        check_against_regression_data
):
    mix, sources = music_mix_and_sources

    sources = list(sources.values())
    mae = nussl.separation.benchmark.MixAsEstimate(
        mix, len(sources))
    estimates = mae()

    evaluator = nussl.evaluation.BSSEvalScale(
        sources, estimates, compute_permutation=False)
    scores = evaluator.evaluate()

    reg_path = os.path.join(REGRESSION_PATH, 'mix_as_estimate.json')
    check_against_regression_data(scores, reg_path)
