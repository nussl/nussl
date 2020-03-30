# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     notebook_metadata_filter: nbsphinx
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   nbsphinx:
#     execute: never
# ---

# Evaluating ideal ratio mask on WHAM!
# =====================================
#
# This recipe evaluates an oracle ideal ratio mask on the `mix_clean`
# and `min` subset in the WHAM dataset. This recipe is annotated 
# as a notebook for documentation but can be run directly
# as a script in `docs/recipes/ideal_ratio_mask.py`.
#
# We evaluate three approaches to constructing the ideal ratio mask:
#
# - Magnitude spectrum approximation
# - Phase sensitive spectrum approximation
# - Truncated phase sensitive spectrum approximation
#
# Imports
# ----------

# +
from nussl import datasets, separation, evaluation
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import logging
import json
import tqdm
import glob
import numpy as np
import termtables

# set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# -

# Setting up
# ----------
#
# Make sure to point `WHAM_ROOT` where you've actually
# built and saved the WHAM dataset. There's a few different
# ways to use ideal ratio masks, so we're going to set those
# up in a dictionary.

# +
WHAM_ROOT = '/home/data/wham/'
NUM_WORKERS = multiprocessing.cpu_count() // 4
OUTPUT_DIR = os.path.expanduser('~/.nussl/recipes/ideal_ratio_mask/')
APPROACHES = {
    'Phase-sensitive spectrum approx.': {
        'kwargs': {
            'range_min': -np.inf, 'range_max':np.inf
        },
        'approach': 'psa',
        'dir': 'psa' 
    },
    'Truncated phase-sensitive approx.': {
        'kwargs': {
            'range_min': 0.0, 'range_max': 1.0
        },
        'approach': 'psa',
        'dir': 'tpsa' 
    },
    'Magnitude spectrum approximation': {
        'kwargs': {},
        'approach': 'msa',
        'dir': 'msa'
    }
}

RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
for key, val in APPROACHES.items():
    _dir = os.path.join(RESULTS_DIR, val['dir'])
    os.makedirs(_dir, exist_ok=True)
# -

# Evaluation
# ----------

# +
test_dataset = datasets.WHAM(WHAM_ROOT, sample_rate=8000, split='tt')

for key, val in APPROACHES.items():
    def separate_and_evaluate(item):
        output_path = os.path.join(
            RESULTS_DIR, val['dir'], f"{item['mix'].file_name}.json")
        separator = separation.benchmark.IdealRatioMask(
            item['mix'], item['sources'], approach=val['approach'],
            mask_type='soft', **val['kwargs'])
        estimates = separator()

        evaluator = evaluation.BSSEvalScale(
            list(item['sources'].values()), estimates, compute_permutation=True)
        scores = evaluator.evaluate()
        with open(output_path, 'w') as f:
            json.dump(scores, f)

    pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
    for i, item in enumerate(tqdm.tqdm(test_dataset)):
        if i == 0:
            separate_and_evaluate(item)
        else:
            pool.submit(separate_and_evaluate, item)
    pool.shutdown(wait=True)

    json_files = glob.glob(f"{RESULTS_DIR}/{val['dir']}/*.json")
    df = evaluation.aggregate_score_files(json_files)

    overall = df.mean()
    print(''.join(['-' for i in range(len(key))]))
    print(key.upper())
    print(''.join(['-' for i in range(len(key))]))
    headers = ["", f"OVERALL (N = {df.shape[0]})", ""]
    metrics = ["SAR", "SDR", "SIR"]
    data = np.array(df.mean()).T

    data = [metrics, data]
    termtables.print(data, header=headers, padding=(0, 1), alignment="ccc")
