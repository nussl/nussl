"""
This recipe trains and evaluates a deep clustering model
on the clean data from the WHAM dataset with 8k. It's divided into 
three big chunks: data preparation, training, and evaluation.
Final output of this script:

┌───────────────────┬────────────────────┬────────────────────┐
│                   │ OVERALL (N = 6000) │                    │
╞═══════════════════╪════════════════════╪════════════════════╡
│        SAR        │        SDR         │        SIR         │
├───────────────────┼────────────────────┼────────────────────┤
│ 11.07829052874508 │ 10.737156798640111 │ 23.704177123014816 │
└───────────────────┴────────────────────┴────────────────────┘

Last run on 3/20/20.
"""
import os
import multiprocessing
import logging
import shutil
import json
import glob
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import optim
import tqdm
import numpy as np
import termtables

import nussl
from nussl import ml, datasets, utils, separation, evaluation


# ----------------------------------------------------
# ------------------- SETTING UP ---------------------
# ----------------------------------------------------

# seed this recipe for reproducibility
utils.seed(0)

# set up logging
logging.basicConfig(	
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',	
    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO) 

# make sure this is set to WHAM root directory
WHAM_ROOT = os.getenv("WHAM_ROOT")
CACHE_ROOT = os.getenv("CACHE_ROOT")
NUM_WORKERS = multiprocessing.cpu_count() // 2
OUTPUT_DIR = os.path.expanduser('~/.nussl/recipes/wham_chimera/run10_1e-4_.5_.5/')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'checkpoints', 'best.model.pth')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 25
MAX_EPOCHS = 100
CACHE_POPULATED = True
LEARNING_RATE = 1e-3
PATIENCE = 5
GRAD_NORM = 1e-2

shutil.rmtree(os.path.join(RESULTS_DIR), ignore_errors=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------------------------------
# ------------------- EVALUATION ---------------------
# ----------------------------------------------------

test_dataset = datasets.WHAM(WHAM_ROOT, sample_rate=8000, split='tt')
# make a deep clustering separator with an empty audio signal initially
# this one will live on gpu and be used in a threadpool for speed
dpcl = separation.deep.DeepClustering(
    nussl.AudioSignal(), num_sources=2, model_path=MODEL_PATH, device='cuda')


def forward_on_gpu(audio_signal):
    # set the audio signal of the object to this item's mix
    dpcl.audio_signal = audio_signal
    features = dpcl.extract_features()
    return features


def separate_and_evaluate(item, features):
    separator = separation.deep.DeepClustering(item['mix'], num_sources=2)
    estimates = separator(features)

    evaluator = evaluation.BSSEvalScale(
        list(item['sources'].values()), estimates, compute_permutation=True)
    scores = evaluator.evaluate()
    output_path = os.path.join(RESULTS_DIR, f"{item['mix'].file_name}.json")
    with open(output_path, 'w') as f:
        json.dump(scores, f)


pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
for i, item in enumerate(tqdm.tqdm(test_dataset)):
    features = forward_on_gpu(item['mix'])
    if i == 0:
        separate_and_evaluate(item, features)
    else:
        pool.submit(separate_and_evaluate, item, features)
pool.shutdown(wait=True)

json_files = glob.glob(f"{RESULTS_DIR}/*.json")
df = evaluation.aggregate_score_files(json_files)

overall = df.mean()
headers = ["", f"OVERALL (N = {df.shape[0]})", ""]
metrics = ["SAR", "SDR", "SIR"]
data = np.array(df.mean()).T

data = [metrics, data]
termtables.print(data, header=headers, padding=(0, 1), alignment="ccc")
