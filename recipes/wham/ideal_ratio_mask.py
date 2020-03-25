"""
This recipe evaluates an oracle ideal ratio mask on the mix_clean 
and min subset in the WHAM dataset using phase sensitive spectrum 
approximation. Output of this script for psa:

┌────────────────────┬────────────────────┬────────────────────┐
│                    │ OVERALL (N = 6000) │                    │
╞════════════════════╪════════════════════╪════════════════════╡
│        SAR         │        SDR         │        SIR         │
├────────────────────┼────────────────────┼────────────────────┤
│ 16.757153130968412 │ 16.43300506403049  │ 28.393687409361203 │
└────────────────────┴────────────────────┴────────────────────┘

Output of this script for msa:

┌────────────────────┬────────────────────┬────────────────────┐
│                    │ OVERALL (N = 6000) │                    │
╞════════════════════╪════════════════════╪════════════════════╡
│        SAR         │        SDR         │        SIR         │
├────────────────────┼────────────────────┼────────────────────┤
│ 13.677899166842302 │ 12.694045978486537 │ 19.854502292474113 │
└────────────────────┴────────────────────┴────────────────────┘

Last run on 3/14/2020
"""
import nussl
from nussl import ml, datasets, utils, separation, evaluation
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
import shutil
import json
import tqdm
import glob
import numpy as np
import termtables

# set up logging
logging.basicConfig(	
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',	
    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO) 

# make sure this is set to WHAM root directory
WHAM_ROOT = os.getenv("WHAM_ROOT")
NUM_WORKERS = multiprocessing.cpu_count() // 4
OUTPUT_DIR = os.path.expanduser('~/.nussl/recipes/ideal_ratio_mask/')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')

# APPROACH, KWARGS = 'psa', {'range_min': -np.inf, 'range_max':np.inf}
APPROACH, KWARGS = 'msa', {}

shutil.rmtree(os.path.join(RESULTS_DIR), ignore_errors=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

test_dataset = datasets.WHAM(WHAM_ROOT, sample_rate=8000, split='tt')


def separate_and_evaluate(item_):
    separator = separation.benchmark.IdealRatioMask(
        item_['mix'], item_['sources'], approach=APPROACH,
        mask_type='soft', **KWARGS)
    estimates = separator()

    evaluator = evaluation.BSSEvalScale(
        list(item_['sources'].values()), estimates, compute_permutation=True)
    scores = evaluator.evaluate()
    output_path = os.path.join(RESULTS_DIR, f"{item_['mix'].file_name}.json")
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
