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

# Evaluating ideal binary mask on WHAM!
# =====================================
#
# This recipe evaluates an oracle ideal binary mask on the `mix_clean`
# and `min` subset in the WHAM dataset. This recipe is annotated 
# as a notebook for documentation but can be run directly
# as a script in `docs/recipes/ideal_binary_mask.py`.
#
# Imports
# -------

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
# built and saved the WHAM dataset.

WHAM_ROOT = '/home/data/wham/'
NUM_WORKERS = multiprocessing.cpu_count() // 4
OUTPUT_DIR = os.path.expanduser('~/.nussl/recipes/ideal_binary_mask/')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Evaluation
# ----------

# +
test_dataset = datasets.WHAM(WHAM_ROOT, sample_rate=8000, split='tt')

def separate_and_evaluate(item):
    separator = separation.benchmark.IdealBinaryMask(
        item['mix'], item['sources'], mask_type='binary')
    estimates = separator()

    evaluator = evaluation.BSSEvalScale(
        list(item['sources'].values()), estimates, compute_permutation=True)
    scores = evaluator.evaluate()
    output_path = os.path.join(RESULTS_DIR, f"{item['mix'].file_name}.json")
    with open(output_path, 'w') as f:
        json.dump(scores, f)


pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
for i, item in enumerate(tqdm.tqdm(test_dataset)):
    if i == 0:
        separate_and_evaluate(item)
    else:
        pool.submit(separate_and_evaluate, item)
pool.shutdown(wait=True)

json_files = glob.glob(f"{RESULTS_DIR}/*.json")
df = evaluation.aggregate_score_files(json_files)

overall = df.mean()
headers = ["", f"OVERALL (N = {df.shape[0]})", ""]
metrics = ["SAR", "SDR", "SIR"]
data = np.array(df.mean()).T

data = [metrics, data]
termtables.print(data, header=headers, padding=(0, 1), alignment="ccc")
