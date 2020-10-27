import pytest
from nussl import efz_utils
import tempfile
import os
import musdb
import zipfile
import scaper
import random
import glob
import nussl
from nussl.datasets import transforms
from nussl import datasets
import numpy as np
import torch
import json


def _unzip(path_to_zip, target_path):
    with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
        zip_ref.extractall(target_path)


fix_dir = os.path.expanduser('~/.nussl/tests/')
os.makedirs(fix_dir, exist_ok=True)
OVERWRITE_REGRESSION_DATA = False


@pytest.fixture(scope="module")
def benchmark_audio():
    audio_files = {}
    keys = ['K0140.wav', 'K0149.wav', 'dev1_female3_inst_mix.wav']
    with tempfile.TemporaryDirectory() as tmp_dir:
        _dir = tmp_dir if fix_dir is None else fix_dir
        for k in keys:
            audio_files[k] = efz_utils.download_audio_file(k, _dir)
        yield audio_files


@pytest.fixture(scope="module")
def musdb_tracks():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _dir = tmp_dir if fix_dir is None else fix_dir
        db = musdb.DB(root=_dir, download=True)
        yield db


@pytest.fixture(scope="module")
def toy_datasets():
    dataset_locations = {}
    keys = ['babywsj_oW0F0H9.zip']
    with tempfile.TemporaryDirectory() as tmp_dir:
        _dir = tmp_dir if fix_dir is None else fix_dir
        for k in keys:
            target_folder = os.path.join(_dir, os.path.splitext(k)[0])
            data = efz_utils.download_benchmark_file(k, _dir)
            _unzip(data, target_folder)
            dataset_locations[k] = target_folder
        yield dataset_locations


@pytest.fixture(scope="module")
def mix_source_folder(toy_datasets):
    wsj_sources = toy_datasets['babywsj_oW0F0H9.zip']
    audio_files = glob.glob(
        f"{wsj_sources}/**/*.wav", recursive=True)
    n_sources = 2
    n_mixtures = 10

    with tempfile.TemporaryDirectory() as tmp_dir:
        _dir = tmp_dir if fix_dir is None else fix_dir
        _dir = os.path.join(_dir, 'mix_source_folder')
        for i in range(n_mixtures):
            sources = []
            for n in range(n_sources):
                path = random.choice(audio_files)
                source = nussl.AudioSignal(path)
                sources.append(source)

            min_length = min([s.signal_length for s in sources])

            for n in range(n_sources):
                output_path = os.path.join(_dir, f's{n}', f'{i}.wav')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                sources[n].truncate_samples(min_length)
                sources[n].write_audio_to_file(output_path)

            mix = sum(sources)
            output_path = os.path.join(_dir, 'mix', f'{i}.wav')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            mix.write_audio_to_file(output_path)
        yield _dir


@pytest.fixture(scope="module")
def scaper_folder(toy_datasets):
    wsj_sources = toy_datasets['babywsj_oW0F0H9.zip']
    fg_path = os.path.join(
        wsj_sources, 'babywsj', 'dev')

    n_sources = 2
    n_mixtures = 10
    duration = 3
    ref_db = -40

    with tempfile.TemporaryDirectory() as tmp_dir:
        _dir = tmp_dir if fix_dir is None else fix_dir
        _dir = os.path.join(_dir, 'scaper')
        os.makedirs(_dir, exist_ok=True)
        for i in range(n_mixtures):
            sc = scaper.Scaper(
                duration, fg_path, fg_path, random_state=i)
            sc.ref_db = ref_db
            sc.sr = 16000

            for j in range(n_sources):
                sc.add_event(
                    label=('choose', []),
                    source_file=('choose', []),
                    source_time=('const', 0),
                    event_time=('const', 0),
                    event_duration=('const', duration),
                    snr=('const', 0),
                    pitch_shift=None,
                    time_stretch=None
                )

            audio_path = os.path.join(_dir, f'{i}.wav')
            jams_path = os.path.join(_dir, f'{i}.jams')
            sc.generate(audio_path, jams_path, save_isolated_events=True)

        yield _dir


@pytest.fixture(scope="module")
def mix_and_sources(scaper_folder):
    dataset = datasets.Scaper(scaper_folder)
    item = dataset[0]
    return item['mix'], item['sources']


@pytest.fixture(scope="module")
def music_mix_and_sources(musdb_tracks):
    dataset = datasets.MUSDB18(
        folder=musdb_tracks.root, download=False,
        transform=transforms.SumSources(
            [['drums', 'bass', 'other']]))
    item = dataset[0]
    return item['mix'], item['sources']


@pytest.fixture(scope="module")
def drum_and_vocals(musdb_tracks):
    dataset = datasets.MUSDB18(
        folder=musdb_tracks.root, download=False)
    item = dataset[0]
    return item['sources']['drums'], item['sources']['vocals']


@pytest.fixture(scope="module")
def bad_scaper_folder(toy_datasets):
    wsj_sources = toy_datasets['babywsj_oW0F0H9.zip']
    fg_path = os.path.join(
        wsj_sources, 'babywsj', 'dev')

    n_sources = 2
    n_mixtures = 10
    duration = 3
    ref_db = -40

    with tempfile.TemporaryDirectory() as tmp_dir:
        _dir = tmp_dir if fix_dir is None else fix_dir
        _dir = os.path.join(_dir, 'bad_scaper')
        os.makedirs(_dir, exist_ok=True)
        for i in range(n_mixtures):
            sc = scaper.Scaper(
                duration, fg_path, fg_path, random_state=i)
            sc.ref_db = ref_db
            sc.sr = 16000

            for j in range(n_sources):
                sc.add_event(
                    label=('choose', []),
                    source_file=('choose', []),
                    source_time=('const', 0),
                    event_time=('const', 0),
                    event_duration=('const', duration),
                    snr=('const', 0),
                    pitch_shift=None,
                    time_stretch=None
                )

            audio_path = os.path.join(_dir, f'{i}.wav')
            jams_path = os.path.join(_dir, f'{i}.jams')
            sc.generate(audio_path, jams_path, save_isolated_events=False)

        yield _dir


@pytest.fixture(scope="module")
def one_item(scaper_folder):
    stft_params = nussl.STFTParams(
        window_length=512,
        hop_length=128
    )
    tfms = transforms.Compose([
        transforms.PhaseSensitiveSpectrumApproximation(),
        transforms.GetAudio(),
        transforms.ToSeparationModel()
    ])
    dataset = nussl.datasets.Scaper(
        scaper_folder, transform=tfms, stft_params=stft_params)
    i = np.random.randint(len(dataset))
    data = dataset[i]
    for k in data:
        # fake a batch dimension
        if torch.is_tensor(data[k]):
            data[k] = data[k].unsqueeze(0)
    yield data


@pytest.fixture(scope="module")
def check_against_regression_data():
    def check(scores, path):
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump(scores, f, indent=4)
        if OVERWRITE_REGRESSION_DATA:
            with open(path, 'w') as f:
                json.dump(scores, f, indent=4)
        else:
            with open(path, 'r') as f:
                reg_scores = json.load(f)
            for key in scores:
                if key not in ['permutation', 'combination']:
                    for metric in scores[key]:
                        if metric in reg_scores[key]:
                            assert np.allclose(
                                scores[key][metric],
                                reg_scores[key][metric],
                                atol=1e-1
                            )
    return check

