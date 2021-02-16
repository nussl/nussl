from nussl.separation.base import DeepMixin, SeparationException
from nussl.separation.base.deep_mixin import OMITTED_TRANSFORMS
from nussl import datasets, ml, separation, evaluation
import nussl
import torch
from torch import optim
import tempfile
import pytest
import os
import numpy as np

fix_dir = 'tests/local/trainer'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCH_LENGTH = 500 if DEVICE == 'cuda' else 1  # use 1 for quick testing, 500 for actual testing
SDR_CUTOFF = 5 if DEVICE == 'cuda' else -20 # use -10 for quick testing, 5 for actual testing

@pytest.fixture(scope="module")
def overfit_model(scaper_folder):
    nussl.utils.seed(0)
    tfms = datasets.transforms.Compose([
        datasets.transforms.PhaseSensitiveSpectrumApproximation(),
        datasets.transforms.MagnitudeWeights(),
        datasets.transforms.ToSeparationModel(),
        datasets.transforms.GetExcerpt(100)
    ])
    dataset = datasets.Scaper(
        scaper_folder, transform=tfms)
    dataset.items = [dataset.items[5]]

    dataloader = torch.utils.data.DataLoader(dataset)

    n_features = dataset[0]['mix_magnitude'].shape[1]
    config = ml.networks.builders.build_recurrent_chimera(
        n_features, 50, 1, True, 0.3, 20, 'sigmoid', 2, 'sigmoid',
        normalization_class='InstanceNorm'
    )
    model = ml.SeparationModel(config)
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    loss_dictionary = {
        'DeepClusteringLoss': {
            'weight': 0.2
        },
        'PermutationInvariantLoss': {
            'args': ['L1Loss'],
            'weight': 0.8
        }
    }

    train_closure = ml.train.closures.TrainClosure(
        loss_dictionary, optimizer, model)

    trainer, _ = ml.train.create_train_and_validation_engines(
        train_closure, device=DEVICE
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        _dir = fix_dir if fix_dir else tmpdir

        ml.train.add_stdout_handler(trainer)
        ml.train.add_validate_and_checkpoint(
            _dir, model, optimizer, dataset, trainer)

        trainer.run(dataloader, max_epochs=1, epoch_length=EPOCH_LENGTH)

        model_path = os.path.join(
            trainer.state.output_folder, 'checkpoints', 'best.model.pth')
        yield model_path, dataset.process_item(dataset.items[0])


def test_deep_mixin(overfit_model):
    model_path, item = overfit_model
    deep_mixin = separation.deep.DeepClustering(
        item['mix'], 2, model_path, mask_type='binary')
    deep_mixin.load_model(model_path)

    deep_mixin.audio_signal = item['mix']
    deep_mixin.channel_dim = -1

    assert not isinstance(deep_mixin.transform, OMITTED_TRANSFORMS)
    if isinstance(deep_mixin.transform, datasets.transforms.Compose):
        for t in deep_mixin.transform.transforms:
            assert not isinstance(t, OMITTED_TRANSFORMS)

    mix_item = {'mix': item['mix']}

    deep_mixin._get_input_data_for_model()

    assert deep_mixin.metadata['stft_params'] == deep_mixin.audio_signal.stft_params

    for key, val in deep_mixin.input_data.items():
        if torch.is_tensor(val):
            assert val.shape[0] == deep_mixin.metadata['num_channels']

    output = deep_mixin.model(deep_mixin.input_data)

    output_tfm = deep_mixin._get_transforms(
        datasets.transforms.MagnitudeWeights())

    output_tfm = deep_mixin._get_transforms(
        datasets.transforms.MagnitudeSpectrumApproximation())

    assert isinstance(
        output_tfm, datasets.transforms.MagnitudeSpectrumApproximation)

    item['mix'].resample(8000)
    deep_mixin.audio_signal = item['mix']
    deep_mixin._get_input_data_for_model()
    assert deep_mixin.audio_signal.sample_rate == deep_mixin.metadata['sample_rate']

    dummy_data = {'one_hot': np.random.rand(100)}
    input_data = deep_mixin._get_input_data_for_model(**dummy_data)

    assert 'one_hot' in input_data


def test_deep_mixin_metadata(overfit_model):
    model_path, item = overfit_model
    deep_mixin = separation.deep.DeepClustering(
        item['mix'], 2, model_path, mask_type='binary')
    deep_mixin.load_model(model_path)

    metadata = deep_mixin.metadata

    deep_mixin.metadata = None
    with pytest.raises(ValueError):
        deep_mixin.get_metadata()

    deep_mixin.metadata = metadata
    assert type(deep_mixin.get_metadata()) == dict
    assert type(deep_mixin.get_metadata(to_str=True)) == str

    result = deep_mixin.get_metadata(for_upload=True)
    keys = ['train_dataset', 'val_dataset']
    for k in keys:
        assert 'folder' not in result[k]
    assert 'trainer.state.epoch_history' not in result
    print(deep_mixin.get_metadata(to_str=True))


def test_separation_deep_clustering(overfit_model):
    model_path, item = overfit_model
    dpcl = separation.deep.DeepClustering(
        item['mix'], 2, model_path, mask_type='binary')
    dpcl.forward() # calls extract_features, for coverage

    item['mix'].write_audio_to_file('tests/local/dpcl_mix.wav')
    sources = item['sources']
    estimates = dpcl()
    for i, e in enumerate(estimates):
        e.write_audio_to_file(f'tests/local/dpcl_overfit{i}.wav')

    evaluator = evaluation.BSSEvalScale(
        list(sources.values()), estimates, compute_permutation=True)
    scores = evaluator.evaluate()

    for key in evaluator.source_labels:
        for metric in ['SI-SDR', 'SI-SIR']:
            _score = scores[key][metric]
            for val in _score:
                assert val > SDR_CUTOFF

    dpcl.model.output_keys = []
    pytest.raises(SeparationException, dpcl.extract_features)


def test_separation_deep_mask_estimation(overfit_model):
    model_path, item = overfit_model
    for mask_type in ['soft', 'binary']:
        dme = separation.deep.DeepMaskEstimation(
            item['mix'], model_path, mask_type=mask_type)

        pytest.raises(SeparationException, dme.confidence)

        item['mix'].write_audio_to_file('tests/local/dme_mix.wav')
        sources = item['sources']
        estimates = dme()
        for i, e in enumerate(estimates):
            e.write_audio_to_file(f'tests/local/dme_overfit{i}.wav')

        evaluator = evaluation.BSSEvalScale(
            list(sources.values()), estimates, compute_permutation=True)
        scores = evaluator.evaluate()

        for key in evaluator.source_labels:
            for metric in ['SI-SDR', 'SI-SIR']:
                _score = scores[key][metric]
                for val in _score:
                    assert val > SDR_CUTOFF

        confidence = dme.confidence()
        dme.model.output_keys = ['mask']
        dme()
        pytest.raises(SeparationException, dme.confidence)
        
        dme.model.output_keys = []
        pytest.raises(SeparationException, dme.run)


@pytest.fixture(scope="module")
def overfit_audio_model(scaper_folder):
    nussl.utils.seed(0)
    tfms = datasets.transforms.Compose([
        datasets.transforms.GetAudio(),
        datasets.transforms.ToSeparationModel(),
        datasets.transforms.GetExcerpt(
            32000, time_dim=1, tf_keys=['mix_audio', 'source_audio'])
    ])
    dataset = datasets.Scaper(
        scaper_folder, transform=tfms)
    dataset.items = [dataset.items[5]]

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1)

    config = ml.networks.builders.build_recurrent_end_to_end(
        256, 256, 64, 'sqrt_hann', 50, 2, 
        True, 0.3, 2, 'sigmoid', num_audio_channels=1, 
        mask_complex=False, rnn_type='lstm', 
        mix_key='mix_audio')

    model = ml.SeparationModel(config)
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_dictionary = {
        'PermutationInvariantLoss': {
            'args': ['SISDRLoss'],
            'weight': 1.0,
            'keys': {'audio': 'estimates', 'source_audio': 'targets'}
        }
    }

    train_closure = ml.train.closures.TrainClosure(
        loss_dictionary, optimizer, model)
    
    trainer, _ = ml.train.create_train_and_validation_engines(
        train_closure, device=DEVICE
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        _dir = os.path.join(fix_dir, 'dae') if fix_dir else tmpdir

        ml.train.add_stdout_handler(trainer)
        ml.train.add_validate_and_checkpoint(
            _dir, model, optimizer, dataset, trainer)
        ml.train.add_progress_bar_handler(trainer)

        trainer.run(dataloader, max_epochs=1, epoch_length=EPOCH_LENGTH)

        model_path = os.path.join(
            trainer.state.output_folder, 'checkpoints', 'best.model.pth')
        yield model_path, dataset.process_item(dataset.items[0])


def test_separation_deep_audio_estimation(overfit_audio_model):
    model_path, item = overfit_audio_model
    dae = separation.deep.DeepAudioEstimation(item['mix'], model_path)

    item['mix'].write_audio_to_file('tests/local/dae_mix.wav')
    sources = item['sources']
    estimates = dae()
    for i, e in enumerate(estimates):
        e.write_audio_to_file(f'tests/local/dae_overfit{i}.wav')

    evaluator = evaluation.BSSEvalScale(
        list(sources.values()), estimates, compute_permutation=True)
    scores = evaluator.evaluate()

    for key in evaluator.source_labels:
        for metric in ['SI-SDR', 'SI-SIR']:
            _score = scores[key][metric]
            for val in _score:
                assert val > SDR_CUTOFF

    dae.model.output_keys = []
    pytest.raises(SeparationException, dae.run)
