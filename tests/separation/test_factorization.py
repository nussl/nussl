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
    acc = copy.deepcopy(sources['drums+bass+other'])

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


def test_ica(
        mix_and_sources,
        check_against_regression_data
):
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
    audio_signals = [
        mix.make_audio_signal_from_channel(ch)
        for ch in range(mix.num_channels)
    ]

    ica = factorization.ICA(audio_signals)
    estimates = ica()

    for e, s in zip(estimates, [a, b]):
        e.to_mono()
        s.to_mono()

    evaluator = nussl.evaluation.BSSEvalScale(
        [a, b], estimates, compute_permutation=True,
        source_labels=['s1', 's2'])
    scores = evaluator.evaluate()

    reg_path = os.path.join(REGRESSION_PATH, 'ica_list.json')
    check_against_regression_data(scores, reg_path)

    ica = factorization.ICA(mix)
    estimates = ica()

    for e in estimates:
        e.to_mono()

    evaluator = nussl.evaluation.BSSEvalScale(
        [a, b], estimates, compute_permutation=True,
        source_labels=['s1', 's2'])
    scores = evaluator.evaluate()

    reg_path = os.path.join(REGRESSION_PATH, 'ica_signal.json')
    check_against_regression_data(scores, reg_path)
