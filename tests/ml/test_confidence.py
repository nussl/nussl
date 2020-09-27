from nussl import ml
import nussl
import torch
import numpy as np 
from sklearn import datasets
import pytest
import copy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.fixture(scope="module")
def simple_sine_data():
    nussl.utils.seed(0)
    folder = 'ignored'

    stft_params = nussl.STFTParams(window_length=256, hop_length=64)
    tfm = nussl.datasets.transforms.Compose([
        nussl.datasets.transforms.PhaseSensitiveSpectrumApproximation(),
        nussl.datasets.transforms.MagnitudeWeights(),
    ])
    tensorize = nussl.datasets.transforms.ToSeparationModel()
    sine_wave_dataset = SineWaves(
        folder, sample_rate=8000, stft_params=stft_params,
        transform=tfm, num_sources=2)
    item = sine_wave_dataset[0]
    tensorized = tensorize(copy.deepcopy(item))

    for key in tensorized:
        if torch.is_tensor(tensorized[key]):
            tensorized[key] = tensorized[key].to(
                DEVICE).float().unsqueeze(0).contiguous()

    return item, tensorized

@pytest.fixture(scope="module")
def simple_model(simple_sine_data):
    item, tensor_data = simple_sine_data

    num_features = 129 # number of frequencies in STFT
    embedding_size = 20 # how many sources to estimate
    activation = ['sigmoid', 'unit_norm'] # activation function for embedding
    num_audio_channels = 1 # number of audio channels

    modules = {
        'mix_magnitude': {},
        'log_spectrogram': {
            'class': 'AmplitudeToDB'
        },
        'normalization': {
            'class': 'BatchNorm',
        },
        'embedding': {
            'class': 'Embedding',
            'args': {
                'num_features': num_features,
                'hidden_size': num_features,
                'embedding_size': embedding_size,
                'activation': activation,
                'num_audio_channels': num_audio_channels,
                'dim_to_embed': [2, 3] # embed the frequency dimension (2) for all audio channels (3)
            }
        },
    }

    connections = [
        ['log_spectrogram', ['mix_magnitude', ]],
        ['normalization', ['log_spectrogram', ]],
        ['embedding', ['normalization', ]],
    ]

    output = ['embedding']

    config = {
        'name': 'SimpleModel',
        'modules': modules,
        'connections': connections,
        'output': output
    }

    model = ml.SeparationModel(config).to(DEVICE)
    untrained = ml.SeparationModel(config).to(DEVICE)
    loss_dictionary = {
        'EmbeddingLoss': {
            'class': 'WhitenedKMeansLoss'
        }
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    closure = ml.train.closures.TrainClosure(loss_dictionary, optimizer, model)

    for i in range(10):
        loss_val = closure(None, tensor_data)
    return model, untrained, item, tensor_data

def test_js_divergence():
    n_samples = 1000
    blobs, _ = datasets.make_blobs(n_samples=n_samples, random_state=8)

    one_component_a = ml.cluster.GaussianMixture(1)
    one_component_b = ml.cluster.GaussianMixture(1)
    two_component = ml.cluster.GaussianMixture(2)

    one_component_a.fit(blobs)
    one_component_b.fit(blobs)
    two_component.fit(blobs)

    confidence_2v1 = ml.confidence.jensen_shannon_divergence(
        one_component_a, two_component)

    confidence_1v1 = ml.confidence.jensen_shannon_divergence(
        one_component_a, one_component_b)

    assert confidence_2v1 > confidence_1v1

def test_get_loud_bins_mask(scaper_folder):
    dataset = nussl.datasets.Scaper(scaper_folder)
    item = dataset[0]

    representation = np.abs(item['mix'].stft())
    mask, _ = ml.confidence._get_loud_bins_mask(0, item['mix'])
    assert representation[mask].sum() == representation.sum()

    mask, _ = ml.confidence._get_loud_bins_mask(100, item['mix'])
    assert not representation[mask]

    mask, _= ml.confidence._get_loud_bins_mask(0, representation=representation)
    assert representation[mask].sum() == representation.sum()

    mask, _ = ml.confidence._get_loud_bins_mask(100,  representation=representation)
    assert not representation[mask]

def test_jensen_shannon_confidence(simple_model):
    model, untrained, item, tensor_data = simple_model

    tr_features = model(tensor_data)['embedding']
    tr_features = tr_features.squeeze(0).transpose(0, 1).data.cpu().numpy()
    
    utr_features = untrained(tensor_data)['embedding']
    utr_features = utr_features.squeeze(0).transpose(0, 1).data.cpu().numpy()

    tr_js_confidence = ml.confidence.jensen_shannon_confidence(
        item['mix'], tr_features, 2
    )

    utr_js_confidence = ml.confidence.jensen_shannon_confidence(
        item['mix'], utr_features, 2
    )
    assert tr_js_confidence > utr_js_confidence

def test_posterior_confidence(simple_model):
    model, untrained, item, tensor_data = simple_model

    tr_features = model(tensor_data)['embedding']
    tr_features = tr_features.squeeze(0).transpose(0, 1).data.cpu().numpy()
    
    utr_features = untrained(tensor_data)['embedding']
    utr_features = utr_features.squeeze(0).transpose(0, 1).data.cpu().numpy()

    tr_confidence = ml.confidence.posterior_confidence(
        item['mix'], tr_features, 2
    )

    utr_confidence = ml.confidence.posterior_confidence(
        item['mix'], utr_features, 2
    )
    assert tr_confidence > utr_confidence

def test_silhouette_confidence(simple_model):
    model, untrained, item, tensor_data = simple_model

    tr_features = model(tensor_data)['embedding']
    tr_features = tr_features.squeeze(0).transpose(0, 1).data.cpu().numpy()
    
    utr_features = untrained(tensor_data)['embedding']
    utr_features = utr_features.squeeze(0).transpose(0, 1).data.cpu().numpy()

    tr_confidence = ml.confidence.silhouette_confidence(
        item['mix'], tr_features, 2
    )

    utr_confidence = ml.confidence.silhouette_confidence(
        item['mix'], utr_features, 2
    )
    assert tr_confidence > utr_confidence

def test_loudness_confidence(simple_model):
    model, untrained, item, tensor_data = simple_model

    tr_features = model(tensor_data)['embedding']
    tr_features = tr_features.squeeze(0).transpose(0, 1).data.cpu().numpy()
    
    utr_features = untrained(tensor_data)['embedding']
    utr_features = utr_features.squeeze(0).transpose(0, 1).data.cpu().numpy()

    tr_confidence = ml.confidence.loudness_confidence(
        item['mix'], tr_features, 2
    )

    utr_confidence = ml.confidence.loudness_confidence(
        item['mix'], utr_features, 2
    )
    assert tr_confidence > utr_confidence

def test_whitened_kmeans_confidence(simple_model):
    model, untrained, item, tensor_data = simple_model

    tr_features = model(tensor_data)['embedding']
    tr_features = tr_features.squeeze(0).transpose(0, 1).data.cpu().numpy()
    
    utr_features = untrained(tensor_data)['embedding']
    utr_features = utr_features.squeeze(0).transpose(0, 1).data.cpu().numpy()

    tr_confidence = ml.confidence.whitened_kmeans_confidence(
        item['mix'], tr_features, 2
    )

    utr_confidence = ml.confidence.whitened_kmeans_confidence(
        item['mix'], utr_features, 2
    )
    assert tr_confidence > utr_confidence

def test_dpcl_classic_confidence(simple_model):
    model, untrained, item, tensor_data = simple_model

    tr_features = model(tensor_data)['embedding']
    tr_features = tr_features.squeeze(0).transpose(0, 1).data.cpu().numpy()
    
    utr_features = untrained(tensor_data)['embedding']
    utr_features = utr_features.squeeze(0).transpose(0, 1).data.cpu().numpy()

    tr_confidence = ml.confidence.dpcl_classic_confidence(
        item['mix'], tr_features, 2
    )

    utr_confidence = ml.confidence.dpcl_classic_confidence(
        item['mix'], utr_features, 2
    )
    assert tr_confidence > utr_confidence


def make_sine_wave(freq, sample_rate, duration):
    dt = 1 / sample_rate
    x = np.arange(0.0, duration, dt)
    x = np.sin(2 * np.pi * freq * x)
    return x

def make_clicks(sample_rate, duration):
    x = np.zeros(sample_rate * duration)
    for i in range(100):
        idx = np.random.randint(x.shape[0])
        x[idx-20:idx+20] = 1
    return x

class SineWaves(nussl.datasets.BaseDataset):
    def __init__(self, *args, num_sources=3, num_frequencies=20, **kwargs):
        self.num_sources = num_sources
        self.frequencies = np.random.choice(
            np.arange(110, 4000, 100), num_frequencies,
            replace=False)

        super().__init__(*args, **kwargs)

    def get_items(self, folder):
        # ignore folder and return a list
        # 100 items in this dataset
        items = list(range(100))
        return items

    def process_item(self, item):
        # we're ignoring items and making
        # sums of random sine waves
        sources = {}
        freqs = []
        freqs = np.random.choice(
            self.frequencies, self.num_sources,
            replace=False)
        for i in range(self.num_sources-1):
            freq = freqs[i]
            _data = make_sine_wave(freq, self.sample_rate, 2)
            # this is a helper function in BaseDataset for
            # making an audio signal from data
            signal = self._load_audio_from_array(_data)
            signal.path_to_input_file = f'{item}.wav'
            sources[f'sine{i}'] = signal * 1 / self.num_sources

        _data = make_clicks(self.sample_rate, 2)
        signal = self._load_audio_from_array(_data)
        signal.path_to_input_file = 'click.wav'
        sources['click'] = signal * 1 / self.num_sources

        mix = sum(sources.values())

        metadata = {
            'frequencies': freqs
        }

        output = {
            'mix': mix,
            'sources': sources,
            'metadata': metadata
        }
        return output
