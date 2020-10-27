from nussl import ml, datasets
import tempfile
from torch import optim
import numpy as np
import logging
import os
import torch

# uncomment if you want to see the trainer/engine logs
logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO
)

fix_dir = 'tests/local/trainer'


def test_create_engine(mix_source_folder):
    # load dataset with transforms
    tfms = datasets.transforms.Compose([
        datasets.transforms.PhaseSensitiveSpectrumApproximation(),
        datasets.transforms.ToSeparationModel(),
        datasets.transforms.Cache(os.path.join(fix_dir, 'cache'))])
    dataset = datasets.MixSourceFolder(
        mix_source_folder, transform=tfms)

    # create the model, based on the first item in the dataset
    # second bit of the shape is the number of features
    n_features = dataset[0]['mix_magnitude'].shape[1]
    mi_config = ml.networks.builders.build_recurrent_mask_inference(
        n_features, 50, 2, True, 0.3, 2, 'softmax',
    )

    model = ml.SeparationModel(mi_config)
    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # dummy function for processing a batch through the model
    def train_batch(engine, data):
        loss = -engine.state.iteration
        return {'loss': loss}

    # building the training and validation engines and running them
    # the validation engine runs within the training engine run
    with tempfile.TemporaryDirectory() as tmpdir:
        _dir = fix_dir if fix_dir else tmpdir
        # _dir = tmpdir
        trainer, validator = ml.train.create_train_and_validation_engines(
            train_batch, train_batch
        )

        # add handlers to engine
        ml.train.add_stdout_handler(trainer, validator)
        ml.train.add_validate_and_checkpoint(_dir, model, optimizer, dataset,
                                             trainer, dataset, validator, 
                                             save_by_epoch=1)
        ml.train.add_tensorboard_handler(_dir, trainer, every_iteration=True)
        ml.train.add_progress_bar_handler(trainer)

        # run engine
        trainer.run(dataset, max_epochs=3)

        assert os.path.exists(trainer.state.output_folder)
        assert os.path.exists(os.path.join(
            trainer.state.output_folder, 'checkpoints', 'latest.model.pth'))
        assert os.path.exists(os.path.join(
            trainer.state.output_folder, 'checkpoints', 'best.model.pth'))
        assert os.path.exists(os.path.join(
            trainer.state.output_folder, 'checkpoints', 'latest.optimizer.pth'))
        assert os.path.exists(os.path.join(
            trainer.state.output_folder, 'checkpoints', 'best.optimizer.pth'))

        for i in range(1, 4):
            assert os.path.exists(os.path.join(
                trainer.state.output_folder, 'checkpoints', 
                f'epoch{i}.model.pth')
            )

        assert len(trainer.state.epoch_history['train/loss']) == 3
        assert len(trainer.state.iter_history['loss']) == 10

        # try resuming
        model_path = os.path.join(
            trainer.state.output_folder, 'checkpoints', 'latest.model.pth')
        optimizer_path = os.path.join(
            trainer.state.output_folder, 'checkpoints', 'latest.optimizer.pth')

        opt_state_dict = torch.load(
            optimizer_path, map_location=lambda storage, loc: storage)
        state_dict = torch.load(
            model_path, map_location=lambda storage, loc: storage)

        optimizer.load_state_dict(opt_state_dict)
        model.load_state_dict(state_dict['state_dict'])

        # make sure the cache got removed in saved transforms bc it's not a portable
        # transform
        for t in state_dict['metadata']['train_dataset']['transforms'].transforms:
            assert not isinstance(t, datasets.transforms.Cache)

        new_trainer, new_validator = (
            ml.train.create_train_and_validation_engines(train_batch)
        )

        # add handlers to engine
        ml.train.add_stdout_handler(new_trainer)
        ml.train.add_validate_and_checkpoint(
            trainer.state.output_folder, model, optimizer, dataset,
            new_trainer)
        ml.train.add_tensorboard_handler(
            trainer.state.output_folder, new_trainer)

        new_trainer.load_state_dict(state_dict['metadata']['trainer.state_dict'])
        assert new_trainer.state.epoch == trainer.state.epoch
        new_trainer.run(dataset, max_epochs=3)


def test_trainer_data_parallel(mix_source_folder):
    # load dataset with transforms
    tfms = datasets.transforms.Compose([
        datasets.transforms.PhaseSensitiveSpectrumApproximation(),
        datasets.transforms.ToSeparationModel()])
    dataset = datasets.MixSourceFolder(
        mix_source_folder, transform=tfms)

    # create the model, based on the first item in the dataset
    # second bit of the shape is the number of features
    n_features = dataset[0]['mix_magnitude'].shape[1]
    mi_config = ml.networks.builders.build_recurrent_mask_inference(
        n_features, 50, 2, True, 0.3, 2, 'softmax',
    )

    model = ml.SeparationModel(mi_config)
    model = torch.nn.DataParallel(model)
    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # dummy function for processing a batch through the model
    def train_batch(engine, data):
        loss = np.random.rand()
        return {'loss': loss}

    # building the training and validation engines and running them
    # the validation engine runs within the training engine run
    with tempfile.TemporaryDirectory() as tmpdir:
        _dir = fix_dir if fix_dir else tmpdir
        trainer, validator = ml.train.create_train_and_validation_engines(
            train_batch, train_batch
        )

        # add handlers to engine
        ml.train.add_stdout_handler(trainer, validator)
        ml.train.add_validate_and_checkpoint(_dir, model, optimizer, dataset,
                                             trainer, dataset, validator)
        ml.train.add_tensorboard_handler(_dir, trainer)

        # run engine
        trainer.run(dataset, max_epochs=3)

        assert os.path.exists(trainer.state.output_folder)
        assert os.path.exists(os.path.join(
            trainer.state.output_folder, 'checkpoints', 'latest.model.pth'))
        assert os.path.exists(os.path.join(
            trainer.state.output_folder, 'checkpoints', 'best.model.pth'))
        assert os.path.exists(os.path.join(
            trainer.state.output_folder, 'checkpoints', 'latest.optimizer.pth'))
        assert os.path.exists(os.path.join(
            trainer.state.output_folder, 'checkpoints', 'best.optimizer.pth'))

        assert len(trainer.state.epoch_history['train/loss']) == 3
        assert len(trainer.state.iter_history['loss']) == 10


def test_cache_dataset(mix_source_folder):
    with tempfile.TemporaryDirectory() as tmpdir:
        tfms = datasets.transforms.Compose([
            datasets.transforms.PhaseSensitiveSpectrumApproximation(),
            datasets.transforms.ToSeparationModel(),
        ])
        chc = datasets.transforms.Cache(
            os.path.join(tmpdir, 'cache'), overwrite=True)

        # no cache
        dataset = datasets.MixSourceFolder(
            mix_source_folder,
            transform=tfms)
        outputs_a = []

        for i in range(len(dataset)):
            outputs_a.append(dataset[i])

        # now add a cache
        tfms.transforms.append(chc)
        dataset = datasets.MixSourceFolder(
            mix_source_folder,
            transform=tfms,
            cache_populated=False)
        assert (
                tfms.transforms[-1].cache.nchunks_initialized == 0)

        ml.train.cache_dataset(dataset)
        assert (
                tfms.transforms[-1].cache.nchunks_initialized == len(dataset))

        # now make sure the cached stuff matches
        dataset.cache_populated = True
        outputs_b = []

        for i in range(len(dataset)):
            outputs_b.append(dataset[i])

        for _data_a, _data_b in zip(outputs_a, outputs_b):
            for key in _data_a:
                if torch.is_tensor(_data_a[key]):
                    assert torch.allclose(_data_a[key], _data_b[key])
                else:
                    assert _data_a[key] == _data_b[key]


def test_cache_dataset_with_dataloader(mix_source_folder):
    with tempfile.TemporaryDirectory() as tmpdir:
        tfms = datasets.transforms.Compose([
            datasets.transforms.PhaseSensitiveSpectrumApproximation(),
            datasets.transforms.ToSeparationModel(),
            datasets.transforms.Cache(
                os.path.join(tmpdir, 'cache'), overwrite=True),
            datasets.transforms.GetExcerpt(400)
        ])
        dataset = datasets.MixSourceFolder(
            mix_source_folder,
            transform=tfms,
            cache_populated=False)

        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=True, batch_size=2)

        ml.train.cache_dataset(dataloader)

        assert (
                tfms.transforms[-2].cache.nchunks_initialized == len(dataset))
