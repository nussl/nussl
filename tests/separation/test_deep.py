from nussl.separation.base import DeepMixin
from nussl.separation.base.deep_mixin import OMITTED_TRANSFORMS
from nussl import datasets, ml, separation, evaluation
import nussl
import torch
from torch import optim
import tempfile
import pytest
import os

fix_dir = 'tests/local/trainer'

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
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

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
        train_closure, device=device
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        _dir = fix_dir if fix_dir else tmpdir

        ml.train.add_stdout_handler(trainer)
        ml.train.add_validate_and_checkpoint(
            _dir, model, optimizer, dataset, trainer)

        trainer.run(dataloader, max_epochs=1, epoch_length=500)

        model_path = os.path.join(
            trainer.state.output_folder, 'checkpoints', 'best.model.pth')
        yield model_path, dataset.process_item(dataset.items[0])

def test_deep_mixin(overfit_model):
    model_path, item = overfit_model
    deep_mixin = DeepMixin()
    deep_mixin.load_model(model_path)

    deep_mixin.audio_signal = item['mix']

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

def test_deep_clustering(overfit_model):
    model_path, item = overfit_model
    dpcl = separation.deep.DeepClustering(
        item['mix'], 2, model_path, mask_type='binary')

    item['mix'].write_audio_to_file('tests/local/dpcl_mix.wav')
    sources = item['sources']
    estimates = dpcl()
    for i, e in enumerate(estimates):
        e.write_audio_to_file(f'tests/local/dpcl_overfit{i}.wav')

    evaluator = evaluation.BSSEvalScale(
        list(sources.values()), estimates, compute_permutation=True)
    scores = evaluator.evaluate()

    for key in evaluator.source_labels:
        for metric in ['SDR', 'SIR']:
            _score = scores[key][metric]  
            for val in _score:
                assert val > 5
