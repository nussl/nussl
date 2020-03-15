import pytest 
from nussl.separation import primitive
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
