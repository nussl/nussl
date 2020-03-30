"""
This recipe trains and evaluates a mask inference model
on the clean data from the WHAM dataset with 8k. It's divided into 
three big chunks: data preparation, training, and evaluation.
Final output of this script:
"""
import nussl
from nussl import ml, datasets, utils, separation, evaluation
import os
import torch
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from torch import optim
import logging
import matplotlib.pyplot as plt
import shutil
import json
import tqdm
import glob
import numpy as np
import termtables

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
NUM_WORKERS = multiprocessing.cpu_count() // 4
OUTPUT_DIR = os.path.expanduser('~/.nussl/recipes/wham_chimera/run14_1e-2_1e3_1')
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
shutil.rmtree(os.path.join(OUTPUT_DIR, 'tensorboard'), ignore_errors=True)

def construct_transforms(cache_location):
    # stft will be 32ms wlen, 8ms hop, sqrt-hann, at 8khz sample rate by default
    tfm = datasets.transforms.Compose([
        datasets.transforms.MagnitudeSpectrumApproximation(), # take stfts and get ibm
        datasets.transforms.MagnitudeWeights(), # get magnitude weights
        datasets.transforms.ToSeparationModel(), # convert to tensors
        datasets.transforms.Cache(cache_location), # up to here gets cached
        datasets.transforms.GetExcerpt(400) # get 400 frame excerpts (3.2 seconds)
    ])
    return tfm

def cache_dataset(_dataset):
    cache_dataloader = torch.utils.data.DataLoader(
        _dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)
    ml.train.cache_dataset(cache_dataloader)
    _dataset.cache_populated = True

tfm = construct_transforms(os.path.join(CACHE_ROOT, 'tr'))
dataset = datasets.WHAM(WHAM_ROOT, split='tr', transform=tfm, 
    cache_populated=CACHE_POPULATED)

tfm = construct_transforms(os.path.join(CACHE_ROOT, 'cv'))
val_dataset = datasets.WHAM(WHAM_ROOT, split='cv', transform=tfm, 
    cache_populated=CACHE_POPULATED)

if not CACHE_POPULATED:
    # cache datasets for speed
    cache_dataset(dataset)
    cache_dataset(val_dataset)

# ----------------------------------------------------
# -------------------- TRAINING ----------------------
# ----------------------------------------------------

# reload after caching
train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
val_sampler = torch.utils.data.sampler.RandomSampler(val_dataset)

dataloader = torch.utils.data.DataLoader(dataset, num_workers=NUM_WORKERS, 
    batch_size=BATCH_SIZE, sampler=train_sampler)
val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=NUM_WORKERS,
    batch_size=BATCH_SIZE, sampler=val_sampler)

n_features = dataset[0]['mix_magnitude'].shape[1]
# builds a baseline model with 4 recurrent layers, 600 hidden units, bidirectional
# and 20 dimensional embedding
config = ml.networks.builders.build_recurrent_chimera(
    n_features, 600, 4, True, 0.3, 20, ['sigmoid', 'unit_norm'], 
    2, ['sigmoid'], normalization_class='BatchNorm'
)
model = ml.SeparationModel(config).to(DEVICE)
logging.info(model)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=PATIENCE)

# set up the loss function
loss_dictionary = {
    'PermutationInvariantLoss': {'args': ['L1Loss'], 'weight': 1e3},
    'DeepClusteringLoss': {'weight': 1.0}
}

# set up closures for the forward and backward pass on one batch
train_closure = ml.train.closures.TrainClosure(
    loss_dictionary, optimizer, model)
val_closure = ml.train.closures.ValidationClosure(
    loss_dictionary, model)

# set up engines for training and validation
trainer, validator = ml.train.create_train_and_validation_engines(
    train_closure, val_closure, device=DEVICE)

# attach handlers for visualizing output and saving the model
ml.train.add_stdout_handler(trainer, validator)
ml.train.add_validate_and_checkpoint(
    OUTPUT_DIR, model, optimizer, dataset, 
    trainer, val_data=val_dataloader, validator=validator)
ml.train.add_tensorboard_handler(OUTPUT_DIR, trainer)

# add a handler to set up patience
@trainer.on(ml.train.ValidationEvents.VALIDATION_COMPLETED)
def step_scheduler(trainer):
    val_loss = trainer.state.epoch_history['validation/loss'][-1]
    scheduler.step(val_loss)

# add a handler to set up gradient clipping
@trainer.on(ml.train.BackwardsEvents.BACKWARDS_COMPLETED)
def clip_gradient(trainer):
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)

# train the model
trainer.run(dataloader, max_epochs=MAX_EPOCHS)

# ----------------------------------------------------
# ------------------- EVALUATION ---------------------
# ----------------------------------------------------

test_dataset = datasets.WHAM(WHAM_ROOT, sample_rate=8000, split='tt')
# make a deep clustering separator with an empty audio signal initially
# this one will live on gpu and be used in a threadpool for speed
dme = separation.deep.DeepMaskEstimation(
    nussl.AudioSignal(), model_path=MODEL_PATH, device='cuda')

def forward_on_gpu(audio_signal):
    # set the audio signal of the object to this item's mix
    dme.audio_signal = audio_signal
    masks = dme.forward()
    return masks

def separate_and_evaluate(item, masks):
    separator = separation.deep.DeepMaskEstimation(item['mix'])
    estimates = separator(masks)

    evaluator = evaluation.BSSEvalScale(
        list(item['sources'].values()), estimates, compute_permutation=True)
    scores = evaluator.evaluate()
    output_path = os.path.join(RESULTS_DIR, f"{item['mix'].file_name}.json")
    with open(output_path, 'w') as f:
        json.dump(scores, f)

pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
for i, item in enumerate(tqdm.tqdm(test_dataset)):
    masks = forward_on_gpu(item['mix'])
    if i == 0:
        separate_and_evaluate(item, masks)
    else:
        pool.submit(separate_and_evaluate, item, masks)
pool.shutdown(wait=True)

json_files = glob.glob(f"{RESULTS_DIR}/*.json")
df = evaluation.aggregate_score_files(json_files)

overall = df.mean()
headers = ["", f"OVERALL (N = {df.shape[0]})", ""]
metrics = ["SAR", "SDR", "SIR"]
data = np.array(df.mean()).T

data = [metrics, data]
termtables.print(data, header=headers, padding=(0, 1), alignment="ccc")
