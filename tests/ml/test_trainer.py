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
            train_batch, train_batch, dataset
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


        assert len(trainer.state.past_iter_history['loss']) == 3 * len(dataset)
        assert len(trainer.state.epoch_history['loss']) == 3
        assert len(trainer.state.iter_history['loss']) == 10

        # try resuming
        model_path = os.path.join(
            trainer.state.output_folder, 'checkpoints', 'best.model.pth')
        optimizer_path = os.path.join(
            trainer.state.output_folder, 'checkpoints', 'best.optimizer.pth')

        opt_state_dict = torch.load(
            optimizer_path, map_location=lambda storage, loc: storage)
        state_dict = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        
        optimizer.load_state_dict(opt_state_dict)
        model.load_state_dict(state_dict['state_dict'])

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
            train_batch, train_batch, dataset
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


        assert len(trainer.state.past_iter_history['loss']) == 3 * len(dataset)
        assert len(trainer.state.epoch_history['loss']) == 3
        assert len(trainer.state.iter_history['loss']) == 10

def test_cache_dataset(mix_source_folder):
    tfms = datasets.transforms.Compose([
        datasets.transforms.PhaseSensitiveSpectrumApproximation(),
        datasets.transforms.ToSeparationModel()
    ])

    dataset = datasets.MixSourceFolder(
        mix_source_folder,
        transform=tfms)
    ml.train.cache_dataset(dataset)
    