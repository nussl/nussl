from nussl import ml, datasets, evaluation
import tempfile
from torch import optim
import numpy as np
import logging
import os
import torch
from matplotlib import pyplot as plt

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO
)

fix_dir = 'tests/local/trainer'


def test_overfit_a(mix_source_folder):
    tfms = datasets.transforms.Compose([
        datasets.transforms.PhaseSensitiveSpectrumApproximation(),
        datasets.transforms.ToSeparationModel(),
        datasets.transforms.Cache('~/.nussl/tests/cache', overwrite=True),
        datasets.transforms.GetExcerpt(400)
    ])
    dataset = datasets.MixSourceFolder(
        mix_source_folder, transform=tfms)

    ml.train.cache_dataset(dataset)
    dataset.cache_populated = True

    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=len(dataset), num_workers=2)

    # create the model, based on the first item in the dataset
    # second bit of the shape is the number of features
    n_features = dataset[0]['mix_magnitude'].shape[1]
    mi_config = ml.networks.builders.build_recurrent_mask_inference(
        n_features, 50, 1, False, 0.0, 2, 'sigmoid',
    )

    model = ml.SeparationModel(mi_config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        epoch_length = 100
    else:
        epoch_length = 10
    model = model.to(device)
    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_dictionary = {
        'L1Loss': {
            'weight': 1.0
        }
    }

    train_closure = ml.train.closures.TrainClosure(
        loss_dictionary, optimizer, model)
    val_closure = ml.train.closures.ValidationClosure(
        loss_dictionary, model)

    with tempfile.TemporaryDirectory() as tmpdir:
        _dir = fix_dir if fix_dir else tmpdir
        os.makedirs(os.path.join(_dir, 'plots'), exist_ok=True)

        trainer, validator = ml.train.create_train_and_validation_engines(
            train_closure, val_closure, device=device
        )

        # add handlers to engine
        ml.train.add_stdout_handler(trainer, validator)
        ml.train.add_validate_and_checkpoint(
            _dir, model, optimizer, dataset,
            trainer, val_data=dataloader, validator=validator)
        ml.train.add_tensorboard_handler(_dir, trainer)

        # run engine
        trainer.run(dataloader, max_epochs=5, epoch_length=epoch_length)

        model_path = os.path.join(
            trainer.state.output_folder, 'checkpoints', 'best.model.pth')
        state_dict = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['state_dict'])

        history = state_dict['metadata']['trainer.state.epoch_history']

        for key in history:
            plt.figure(figsize=(10, 4))
            plt.title(f"epoch:{key}")
            plt.plot(np.array(history[key]).reshape(-1, ))
            plt.savefig(os.path.join(
                trainer.state.output_folder, 'plots',
                f"epoch:{key.replace('/', ':')}.png"))

