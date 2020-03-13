import pytest
import nussl
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
        [a, b], estimates, compute_permutation=True)
    scores = evaluator.evaluate()

    reg_path = os.path.join(REGRESSION_PATH, 'spatial_clustering.json')
    check_against_regression_data(scores, reg_path)

