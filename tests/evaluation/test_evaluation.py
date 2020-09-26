import nussl
import pytest
from nussl.core.masks import SoftMask, BinaryMask
import numpy as np
from nussl.evaluation.evaluation_base import AudioSignalListMismatchError
import torch
import json
import tempfile
import os
import glob


@pytest.fixture(scope='module')
def estimated_and_true_sources(musdb_tracks):
    i = np.random.randint(len(musdb_tracks))
    track = musdb_tracks[i]
    mixture = nussl.AudioSignal(
        audio_data_array=track.audio,
        sample_rate=track.rate)
    mixture.stft()

    stems = track.stems
    oracle_sources = []
    random_sources = []
    true_sources = []
    random_masks = []
    oracle_masks = []
    keys = []

    for k, v in sorted(track.sources.items(), key=lambda x: x[1].stem_id):
        true_sources.append(nussl.AudioSignal(
            audio_data_array=stems[v.stem_id],
            sample_rate=track.rate
        ))
        keys.append(k)

        mask_data = np.random.rand(*mixture.stft_data.shape)
        random_mask = SoftMask(mask_data)
        random_source = mixture.apply_mask(random_mask)
        random_source.istft(truncate_to_length=mixture.signal_length)

        random_sources.append(random_source)
        random_masks.append(random_mask)

        source_stft = true_sources[-1].stft()

        mask_data = (
                (np.abs(source_stft) + 1e-8) /
                (np.maximum(np.abs(mixture.stft_data), np.abs(source_stft)) + 1e-8)
        )
        oracle_mask = SoftMask(mask_data)
        oracle_source = mixture.apply_mask(oracle_mask)
        oracle_source.istft(truncate_to_length=mixture.signal_length)

        oracle_sources.append(oracle_source)
        oracle_masks.append(oracle_mask)

    yield {
        'oracle': oracle_sources,
        'random': random_sources,
        'true': true_sources,
        'keys': keys,
        'oracle_masks': oracle_masks,
        'random_masks': random_masks,
    }


def fake_preprocess_a(self):
    references = np.random.rand(44100, 2, 4)
    estimates = np.random.rand(44100, 2, 4)
    return references, estimates


def fake_preprocess_b(self):
    references = np.random.rand(44100, 2, 2)
    estimates = np.random.rand(44100, 2, 4)
    return references, estimates


def fake_evaluate_helper(self, references, estimates):
    n_sources = references.shape[-1]
    n_channels = references.shape[-2]
    scores = []
    for i in range(n_sources):
        score = {
            'metric1': [np.random.rand()] * n_channels,
            'metric2': [np.random.rand()] * n_channels,
            'metric3': [np.random.rand()] * n_channels,
        }
        scores.append(score)
    return scores


def test_evaluation_base(estimated_and_true_sources):
    true_sources = estimated_and_true_sources['true']
    estimated_sources = estimated_and_true_sources['random']
    keys = estimated_and_true_sources['keys']

    default_source_labels = [f'source_{i}' for i in range(len(true_sources))]
    evaluator = nussl.evaluation.EvaluationBase(true_sources, estimated_sources)
    assert (
            evaluator.source_labels == default_source_labels)

    for k, t in zip(keys, true_sources):
        t.path_to_input_file = k

    evaluator = nussl.evaluation.EvaluationBase(true_sources, estimated_sources)
    assert evaluator.source_labels == keys

    source_labels = [f'mysource_{i}' for i in range(len(keys))]
    evaluator = nussl.evaluation.EvaluationBase(
        true_sources, estimated_sources, source_labels=source_labels)
    assert evaluator.source_labels == source_labels

    source_labels = [f'mysource_{i}' for i in range(len(keys) - 2)]
    evaluator = nussl.evaluation.EvaluationBase(
        true_sources, estimated_sources, source_labels=source_labels)
    assert evaluator.source_labels == source_labels + keys[2:]

    for k, t in zip(keys, true_sources):
        t.path_to_input_file = None
    evaluator = nussl.evaluation.EvaluationBase(
        true_sources, estimated_sources, source_labels=source_labels)
    assert evaluator.source_labels == source_labels + default_source_labels[2:]

    assert evaluator.scores == {}


def test_evaluation_run(estimated_and_true_sources, monkeypatch):
    monkeypatch.setattr(
        nussl.evaluation.EvaluationBase, 'preprocess', fake_preprocess_a)
    monkeypatch.setattr(
        nussl.evaluation.EvaluationBase, 'evaluate_helper', fake_evaluate_helper)

    true_sources = estimated_and_true_sources['true']
    estimated_sources = estimated_and_true_sources['random']
    keys = estimated_and_true_sources['keys']

    for k, t in zip(keys, true_sources):
        t.path_to_input_file = k

    evaluator = nussl.evaluation.EvaluationBase(true_sources, estimated_sources)
    candidates = evaluator.get_candidates()
    assert len(candidates[0]) == 1
    assert len(candidates[1]) == 1
    evaluator.evaluate()
    check_scores(evaluator)

    evaluator = nussl.evaluation.EvaluationBase(true_sources, estimated_sources,
                                                compute_permutation=True)
    candidates = evaluator.get_candidates()
    # should be 4 choose 2
    assert len(candidates[0]) == 1
    assert len(candidates[1]) == 24
    evaluator.evaluate()
    check_scores(evaluator)

    evaluator = nussl.evaluation.EvaluationBase(true_sources, estimated_sources,
                                                compute_permutation=True,
                                                best_permutation_key='metric2')
    candidates = evaluator.get_candidates()
    # should be 1 * 4! = 24
    assert len(candidates[0]) == 1
    assert len(candidates[1]) == 24
    evaluator.evaluate()
    check_scores(evaluator)

    monkeypatch.setattr(
        nussl.evaluation.EvaluationBase, 'preprocess', fake_preprocess_b)

    evaluator = nussl.evaluation.EvaluationBase(true_sources[:2], estimated_sources,
                                                compute_permutation=True)
    candidates = evaluator.get_candidates()
    # should be (4 choose 2) * 2! = 12
    assert len(candidates[0]) == 6
    assert len(candidates[1]) == 2
    evaluator.evaluate()
    check_scores(evaluator)


def check_scores(evaluator):
    assert evaluator.scores is not None
    assert isinstance(evaluator.scores, dict)

    assert 'combination' in evaluator.scores.keys()
    assert 'permutation' in evaluator.scores.keys()

    for source_label in evaluator.source_labels:
        assert source_label in evaluator.scores


def test_bss_evaluation_base(estimated_and_true_sources, monkeypatch):
    monkeypatch.setattr(
        nussl.evaluation.EvaluationBase, 'evaluate_helper', fake_evaluate_helper)

    true_sources = estimated_and_true_sources['true']
    estimated_sources = estimated_and_true_sources['random']
    keys = estimated_and_true_sources['keys']

    for k, t in zip(keys, true_sources):
        t.path_to_input_file = k

    evaluator = nussl.evaluation.BSSEvaluationBase(
        true_sources, estimated_sources)
    references, estimates = evaluator.preprocess()

    n_samples = true_sources[0].signal_length
    n_channels = true_sources[0].num_channels
    n_sources = len(true_sources)

    assert references.shape == (n_samples, n_channels, n_sources)
    assert estimates.shape == (n_samples, n_channels, n_sources)


def test_bss_eval_v4(estimated_and_true_sources):
    true_sources = estimated_and_true_sources['true']
    estimated_sources = estimated_and_true_sources['random']
    keys = estimated_and_true_sources['keys']
    for k, t in zip(keys, true_sources):
        t.path_to_input_file = k

    evaluator = nussl.evaluation.BSSEvalV4(
        true_sources, estimated_sources)
    references, estimates = evaluator.preprocess()
    scores = evaluator.evaluate_helper(references, estimates)
    assert isinstance(scores, list)

    random_scores = evaluator.evaluate()
    check_scores(evaluator)

    estimated_sources = estimated_and_true_sources['oracle']
    evaluator = nussl.evaluation.BSSEvalV4(
        true_sources, estimated_sources)
    oracle_scores = evaluator.evaluate()

    # the oracle score should beat the random score by a lot on average
    # for SDR and SIR

    for key in evaluator.source_labels:
        for metric in ['SDR', 'SIR']:
            _oracle = oracle_scores[key][metric]
            _random = random_scores[key][metric]

            assert np.alltrue(_oracle > _random)


def test_scale_bss_eval(estimated_and_true_sources):
    true_sources = estimated_and_true_sources['true']
    estimated_sources = estimated_and_true_sources['oracle']
    random_sources = estimated_and_true_sources['random']

    evaluator = nussl.evaluation.BSSEvalScale(
        true_sources, estimated_sources)
    references, estimates = evaluator.preprocess()
    _references = references[:, 0, :]
    _estimates = estimates[:, 0, :]
    mixture = sum(true_sources).audio_data.T
    mixture -= mixture.mean(axis=0)

    tSISDR, tSISIR, tSISAR, tSDSDR, tSNR, tSRR, tSDRi, tSDSDRi, tSNRi, _, _, _ = (
        nussl.evaluation.scale_bss_eval(
            _references, _estimates[..., 0], mixture[..., 0], 0,
        )
    )

    evaluator = nussl.evaluation.BSSEvalScale(
        true_sources, random_sources)
    references, estimates = evaluator.preprocess()
    _references = references[:, 0, :]
    _estimates = estimates[:, 0, :]

    rSISDR, rSISIR, rSISAR, rSDSDR, rSNR, rSRR, tSDRi, tSDSDRi, tSNRi, _, _, _ = (
        nussl.evaluation.scale_bss_eval(
            _references, _estimates[..., 0], mixture[..., 0], 0,
        )
    )

    assert tSISDR > rSISDR
    assert tSISIR > rSISIR
    assert tSDSDR > rSDSDR
    assert tSNR > rSNR


def test_bss_eval_scale(estimated_and_true_sources):
    compute_sir_sar = [True, False]

    for _compute_sir_sar in compute_sir_sar:
        with tempfile.TemporaryDirectory() as tmpdir:
            true_sources = estimated_and_true_sources['true']
            estimated_sources = estimated_and_true_sources['random']
            keys = estimated_and_true_sources['keys']

            for k, t in zip(keys, true_sources):
                t.path_to_input_file = k

            evaluator = nussl.evaluation.BSSEvalScale(
                true_sources, estimated_sources, 
                compute_sir_sar=_compute_sir_sar)
            references, estimates = evaluator.preprocess()
            scores = evaluator.evaluate_helper(
                references, estimates, 
                compute_sir_sar=_compute_sir_sar)
            assert isinstance(scores, list)

            random_scores = evaluator.evaluate()
            check_scores(evaluator)

            estimated_sources = estimated_and_true_sources['oracle']

            evaluator = nussl.evaluation.BSSEvalScale(
                true_sources, estimated_sources,
                compute_sir_sar=_compute_sir_sar)
            oracle_scores = evaluator.evaluate()

            # the oracle score should beat the random score by a lot on average
            # for SDR and SIR
            for key in evaluator.source_labels:
                for metric in ['SI-SDR', 'SNR', 'SD-SDR']:
                    _oracle = oracle_scores[key][metric]
                    _random = random_scores[key][metric]

                    assert np.alltrue(_oracle > _random)

            save_scores(tmpdir, oracle_scores, f'oracle.json')
            save_scores(tmpdir, random_scores, f'random.json')

            check_aggregate(tmpdir)
            check_associate_metrics(tmpdir)


def save_scores(directory, scores, name):
    with open(os.path.join(directory, f'{name}.json'), 'w') as f:
        json.dump(scores, f)


def check_aggregate(directory):
    json_files = glob.glob(f"{directory}/*.json")
    df = nussl.evaluation.aggregate_score_files(json_files)

    nussl.separation.deep.DeepMaskEstimation(nussl.AudioSignal())

    report_card = nussl.evaluation.report_card(df, 'Testing notes', decimals=5)

    assert 'Testing notes' in report_card
    report_card_overall = nussl.evaluation.report_card(
        df, 'Testing notes', report_each_source=False)
    print(report_card_overall)
    assert len(report_card_overall) < len(report_card)


def check_associate_metrics(directory):
    json_files = glob.glob(f"{directory}/*.json")
    df = nussl.evaluation.aggregate_score_files(json_files)

    n_sources = 2
    duration = 3
    sample_rate = 44100
    min_freq, max_freq = 110, 1000

    def make_mix(dataset, i):
        sources = {}
        freqs = []
        for i in range(n_sources):
            freq = np.random.randint(min_freq, max_freq)
            freqs.append(freq)
            dt = 1 / sample_rate
            source_data = np.arange(0.0, duration, dt)
            source_data = np.sin(2 * np.pi * freq * source_data)
            source_signal = dataset._load_audio_from_array(
                audio_data=source_data, sample_rate=sample_rate)
            sources[f'sine{i}'] = source_signal * 1 / n_sources
        mix = sum(sources.values())
        output = {
            'mix': mix,
            'sources': sources,
            'metadata': {
                'frequencies': freqs    
            }    
        }
        return output

    dataset = nussl.datasets.OnTheFly(make_mix, 10)

    n_features = 257
    mi_config = nussl.ml.networks.builders.build_recurrent_mask_inference(
        n_features, 50, 1, False, 0.0, 2, 'sigmoid',
    )
    model = nussl.ml.SeparationModel(mi_config)

    model = nussl.evaluation.associate_metrics(model, df, dataset)

    assert 'evaluation' in model.metadata.keys()

    assert 'source' not in model.metadata['evaluation'].keys()
    assert 'file' not in model.metadata['evaluation'].keys()

    stats_keys = ['mean', 'median', 'std']
    for metric in model.metadata['evaluation'].values():
        assert all(s in metric.keys() for s in stats_keys)

    assert 'test_dataset' in model.metadata.keys()

    sm_keys = ['name', 'stft_params', 'sample_rate',
               'num_channels', 'folder', 'transforms']

    assert all(k in model.metadata['test_dataset'].keys() for k in sm_keys)



def test_eval_permutation(estimated_and_true_sources):
    true_sources = estimated_and_true_sources['true'][:2]
    estimated_sources = estimated_and_true_sources['oracle'][:2]
    keys = estimated_and_true_sources['keys']
    for k, t in zip(keys, true_sources):
        t.path_to_input_file = k

    evaluator = nussl.evaluation.BSSEvalV4(
        true_sources, estimated_sources[::-1],
        compute_permutation=True)
    scores = evaluator.evaluate()
    assert scores['permutation'] == (1, 0)

    true_sources = estimated_and_true_sources['true']
    estimated_sources = estimated_and_true_sources['oracle']

    evaluator = nussl.evaluation.BSSEvalScale(
        true_sources, estimated_sources[::-1],
        compute_permutation=True)
    scores = evaluator.evaluate()
    assert scores['permutation'] == (3, 2, 1, 0)

    oracle_masks = estimated_and_true_sources['oracle_masks']
    estimated_masks = estimated_and_true_sources['oracle_masks'][::-1]

    oracle_masks = [o.mask_to_binary() for o in oracle_masks]
    estimated_masks = [r.mask_to_binary() for r in estimated_masks]

    nussl.evaluation.PrecisionRecallFScore(
        oracle_masks, estimated_masks)
    scores = evaluator.evaluate()
    assert scores['permutation'] == (3, 2, 1, 0)


def test_eval_precision_recall_fscore(estimated_and_true_sources):
    oracle_masks = estimated_and_true_sources['oracle_masks']
    random_masks = estimated_and_true_sources['random_masks']

    pytest.raises(ValueError,
                  nussl.evaluation.PrecisionRecallFScore, oracle_masks, random_masks
                  )

    random_extra_mask = [BinaryMask(np.random.rand(100, 10, 2) > .5)]

    oracle_masks = [o.mask_to_binary() for o in oracle_masks]
    random_masks = [r.mask_to_binary() for r in random_masks]

    pytest.raises(ValueError,
                  nussl.evaluation.PrecisionRecallFScore,
                  oracle_masks + random_extra_mask,
                  random_masks + random_extra_mask
                  )

    evaluator = nussl.evaluation.PrecisionRecallFScore(
        oracle_masks[0], random_masks[0])
    references, estimates = evaluator.preprocess()

    shape = (
        oracle_masks[0].mask.shape[0] * oracle_masks[0].mask.shape[1],
        oracle_masks[0].num_channels, 1)

    assert references.shape == shape
    assert estimates.shape == shape

    evaluator = nussl.evaluation.PrecisionRecallFScore(
        oracle_masks, random_masks)
    references, estimates = evaluator.preprocess()

    shape = (
        oracle_masks[0].mask.shape[0] * oracle_masks[0].mask.shape[1],
        oracle_masks[0].num_channels, len(oracle_masks))

    assert references.shape == shape
    assert estimates.shape == shape

    scores = evaluator.evaluate()
    check_scores(evaluator)
