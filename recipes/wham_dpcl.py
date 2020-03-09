"""
This recipe trains and evaluates a deep clustering model
on the clean data from the WHAM dataset with 8k.
"""
from nussl import ml, datasets, utils
import os
import torch
from multiprocessing import cpu_count
from torch import optim
import logging
import matplotlib.pyplot as plt

# seed this recipe for reproducibility
utils.seed(0)

# set up logging
logging.basicConfig(	
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',	
    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO) 

# make sure this is set to WHAM root directory
WHAM_ROOT = os.getenv("WHAM_ROOT")
CACHE_ROOT = os.getenv("CACHE_ROOT")
NUM_WORKERS = cpu_count() // 2
OUTPUT_DIR = os.path.expanduser('~/.nussl/recipes/wham_dpcl/')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 20
MAX_EPOCHS = 80

def construct_transforms(cache_location):
    # stft will be 32ms wlen, 8ms hop, sqrt-hann, at 8khz sample rate by default
    tfm = datasets.transforms.Compose([
        datasets.transforms.MagnitudeSpectrumApproximation(), # take stfts and get ibm
        datasets.transforms.MagnitudeWeights(), # get magnitude weights
        datasets.transforms.ToSeparationModel(), # convert to tensors
        datasets.transforms.Cache(cache_location, overwrite=True),
        datasets.transforms.GetExcerpt(400) # get 400 frame excerpts (3.2 seconds)
    ])
    return tfm

def cache_dataset(_dataset):
    cache_dataloader = torch.utils.data.DataLoader(
        _dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)
    ml.train.cache_dataset(cache_dataloader)
    _dataset.cache_populated = True

tfm = construct_transforms(os.path.join(CACHE_ROOT, 'tr'))
dataset = datasets.WHAM(WHAM_ROOT, split='tr', transform=tfm, cache_populated=False)

tfm = construct_transforms(os.path.join(CACHE_ROOT, 'cv'))
val_dataset = datasets.WHAM(WHAM_ROOT, split='cv', transform=tfm, cache_populated=False)

# cache datasets for speed
cache_dataset(dataset)
cache_dataset(val_dataset)

# reload after caching
dataloader = torch.utils.data.DataLoader(dataset, num_workers=NUM_WORKERS, 
    batch_size=BATCH_SIZE)
val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=NUM_WORKERS,
    batch_size=BATCH_SIZE)

n_features = dataset[0]['mix_magnitude'].shape[1]
# builds a baseline model with 4 recurrent layers, 600 hidden units, bidirectional
# and 20 dimensional embedding
config = ml.networks.builders.build_recurrent_dpcl(
    n_features, 600, 4, True, 0.3, 20, ['sigmoid'])
model = ml.SeparationModel(config).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

# set up the loss function
loss_dictionary = {
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

@trainer.on(ml.train.ValidationEvents.VALIDATION_COMPLETED)
def visualize_grad_norm(trainer):
    utils.visualize_gradient_flow(model.named_parameters())
    plt.savefig(os.path.join(OUTPUT_DIR, 'grad.png'))

# train the model
trainer.run(dataloader, max_epochs=MAX_EPOCHS)