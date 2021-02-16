import pytest
import nussl
from nussl.separation import SeparationException
import numpy as np
import os

REGRESSION_PATH = 'tests/separation/regression/spatial/'
os.makedirs(REGRESSION_PATH, exist_ok=True)


def test_spatial_clustering(mix_and_sources, check_against_regression_data):
    nussl.utils.seed(0)
    mix, sources = mix_and_sources
    sources = list(sources.values())

    a = nussl.mixing.pan_audio_signal(sources[0], -35)
    a_delays = [np.random.randint(1, 200) for _ in range(a.num_channels)]
    a = nussl.mixing.delay_audio_signal(a, a_delays)

    b = nussl.mixing.pan_audio_signal(sources[1], 15)
    b_delays = [np.random.randint(1, 200) for _ in range(b.num_channels)]
    b = nussl.mixing.delay_audio_signal(b, b_delays)

    mix = a + b
    spcl = nussl.separation.spatial.SpatialClustering(mix, num_sources=2)
    estimates = spcl()
    for e in estimates:
        e.to_mono()
    for s in [a, b]:
        s.to_mono()

    evaluator = nussl.evaluation.BSSEvalScale(
        [a, b], estimates, compute_permutation=True, 
        source_labels=['s1', 's2'])
    scores = evaluator.evaluate()

    reg_path = os.path.join(REGRESSION_PATH, 'spatial_clustering.json')
    check_against_regression_data(scores, reg_path, atol=1e-1)


def test_duet(mix_and_sources, check_against_regression_data):
    nussl.utils.seed(0)
    mix, sources = mix_and_sources
    sources = list(sources.values())

    a = nussl.mixing.pan_audio_signal(sources[0], -35)
    a_delays = [np.random.randint(1, 20) for _ in range(a.num_channels)]
    a = nussl.mixing.delay_audio_signal(a, a_delays)

    b = nussl.mixing.pan_audio_signal(sources[1], 35)
    b_delays = [np.random.randint(1, 20) for _ in range(b.num_channels)]
    b = nussl.mixing.delay_audio_signal(b, b_delays)

    mix = a + b
    duet = nussl.separation.spatial.Duet(mix, num_sources=2)
    estimates = duet()

    for e in estimates:
        e.to_mono()

    for s in [a, b]:
        s.to_mono()

    evaluator = nussl.evaluation.BSSEvalScale(
        [a, b], estimates, compute_permutation=True,
        source_labels=['s1', 's2'])
    scores = evaluator.evaluate()

    reg_path = os.path.join(REGRESSION_PATH, 'duet.json')
    check_against_regression_data(scores, reg_path, atol=1e-1)


def test_projet(
        drum_and_vocals,
        check_against_regression_data
):
    nussl.utils.seed(0)

    drum, vocals = drum_and_vocals
    drum = nussl.mixing.pan_audio_signal(drum, 30)
    vocals = nussl.mixing.pan_audio_signal(vocals, -30)

    mix = drum + vocals

    sep = nussl.separation.spatial.Projet(mix, 2)
    estimates = sep()

    evaluator = nussl.evaluation.BSSEvalScale(
        [drum, vocals], estimates, compute_permutation=True)
    scores = evaluator.evaluate()

    reg_path = os.path.join(REGRESSION_PATH, 'projet_pan.json')
    check_against_regression_data(scores, reg_path)

    # now put some delays
    delays = [np.random.randint(1, 20) for _ in range(drum.num_channels)]
    drum = nussl.mixing.delay_audio_signal(drum, delays)
    delays = [np.random.randint(1, 20) for _ in range(vocals.num_channels)]
    vocals = nussl.mixing.delay_audio_signal(vocals, delays)

    mix = drum + vocals

    sep = nussl.separation.spatial.Projet(mix, 2)
    estimates = sep()

    evaluator = nussl.evaluation.BSSEvalScale(
        [drum, vocals], estimates, compute_permutation=True)
    scores = evaluator.evaluate()

    reg_path = os.path.join(REGRESSION_PATH, 'projet_delay.json')
    check_against_regression_data(scores, reg_path)

    # now do some initialization of the PSDs

    ft2d = nussl.separation.primitive.FT2D(mix)
    ft2d_estimates = ft2d()

    pytest.raises(SeparationException, nussl.separation.spatial.Projet,
                  mix, 2, estimates=[ft2d_estimates[0]])

    sep = nussl.separation.spatial.Projet(mix, 2, estimates=ft2d_estimates)
    estimates = sep()

    evaluator = nussl.evaluation.BSSEvalScale(
        [drum, vocals], estimates, compute_permutation=True)
    scores = evaluator.evaluate()

    reg_path = os.path.join(REGRESSION_PATH, 'projet_with_init.json')
    check_against_regression_data(scores, reg_path)
