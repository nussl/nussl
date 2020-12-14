import pytest
import nussl
from pretty_midi import PrettyMIDI, Instrument, Note
from nussl.core import constants
import os
import numpy as np
import yaml
from yaml import Dumper
from nussl.datasets.base_dataset import DataSetException
from nussl.datasets import transforms
import tempfile
import shutil


def test_dataset_hook_musdb18(musdb_tracks):
    dataset = nussl.datasets.MUSDB18(
        folder=musdb_tracks.root, download=True)

    data = dataset[0]
    track = musdb_tracks[0]
    stems = track.stems

    assert np.allclose(data['mix'].audio_data, track.audio.T)

    for k, v in sorted(track.sources.items(), key=lambda x: x[1].stem_id):
        assert np.allclose(
            data['sources'][k].audio_data, stems[v.stem_id].T)

    dataset = nussl.datasets.MUSDB18(
        folder=None, download=False)
    assert dataset.folder == os.path.join(
        constants.DEFAULT_DOWNLOAD_DIRECTORY, 'musdb18')

    for k in data['sources']:
        assert k.split('::')[0] in data['metadata']['labels']


def test_dataset_hook_mix_source_folder(mix_source_folder):
    dataset = nussl.datasets.MixSourceFolder(mix_source_folder)
    data = dataset[0]

    _sources = [data['sources'][k] for k in data['sources']]
    assert np.allclose(sum(_sources).audio_data, data['mix'].audio_data)

    for k in data['sources']:
        assert k.split('::')[0] in data['metadata']['labels']

    dataset = nussl.datasets.MixSourceFolder(mix_source_folder, make_mix=True)
    data = dataset[0]

    _sources = [data['sources'][k] for k in data['sources']]
    assert np.allclose(sum(_sources).audio_data, data['mix'].audio_data)

    for k in data['sources']:
        assert k.split('::')[0] in data['metadata']['labels']


def test_dataset_hook_scaper_folder(scaper_folder):
    dataset = nussl.datasets.Scaper(scaper_folder)
    data = dataset[0]

    _sources = [data['sources'][k] for k in data['sources']]
    assert np.allclose(sum(_sources).audio_data, data['mix'].audio_data)

    for k in data['sources']:
        assert k.split('::')[0] in data['metadata']['labels']

    # make sure SumSources transform works
    tfm = transforms.SumSources(
        [['050', '051']],
        group_names=['both'],
    )

    data = tfm(data)

    for k in data['sources']:
        assert k.split('::')[0] in data['metadata']['labels']

    _sources = [data['sources'][k] for k in data['sources']]
    assert np.allclose(sum(_sources).audio_data, data['mix'].audio_data)


def test_dataset_hook_bad_scaper_folder(bad_scaper_folder):
    pytest.raises(
        DataSetException, nussl.datasets.Scaper, bad_scaper_folder)


def test_dataset_hook_wham(benchmark_audio):
    # make a fake wham dir structure
    audio = nussl.AudioSignal(benchmark_audio['K0140.wav'])
    with tempfile.TemporaryDirectory() as tmpdir:
        for wav_folder in ['wav8k', 'wav16k']:
            sr = 8000 if wav_folder == 'wav8k' else 16000
            for mode in ['min', 'max']:
                for split in ['tr', 'cv', 'tt']:
                    for key, val in nussl.datasets.WHAM.MIX_TO_SOURCE_MAP.items():
                        parent = os.path.join(
                            tmpdir, wav_folder, mode, split)
                        mix_path = os.path.join(parent, key)
                        os.makedirs(mix_path, exist_ok=True)
                        audio.resample(sr)
                        audio.write_audio_to_file(
                            os.path.join(mix_path, '0.wav'))
                        for x in val:
                            source_path = os.path.join(parent, x)
                            os.makedirs(source_path, exist_ok=True)
                            audio.write_audio_to_file(
                                os.path.join(source_path, '0.wav'))

        for mode in ['min', 'max']:
            for split in ['tr', 'cv', 'tt']:
                for sr in [8000, 16000]:
                    wham = nussl.datasets.WHAM(
                        root=tmpdir, mix_folder='mix_clean', mode=mode,
                        split=split, sample_rate=sr)
                    output = wham[0]
                    assert output['metadata']['labels'] == ['s1', 's2']

                    wham = nussl.datasets.WHAM(
                        root=tmpdir, mix_folder='mix_both', mode=mode,
                        split=split, sample_rate=sr)
                    output = wham[0]
                    assert output['metadata']['labels'] == [
                        's1', 's2', 'noise']

                    wham = nussl.datasets.WHAM(
                        root=tmpdir, mix_folder='mix_single', mode=mode,
                        split=split, sample_rate=sr)
                    output = wham[0]
                    assert output['metadata']['labels'] == ['s1']

        pytest.raises(DataSetException, nussl.datasets.WHAM,
                      tmpdir, mix_folder='not matching')

        pytest.raises(DataSetException, nussl.datasets.WHAM,
                      tmpdir, mode='not matching')

        pytest.raises(DataSetException, nussl.datasets.WHAM,
                      tmpdir, split='not matching')

        pytest.raises(DataSetException, nussl.datasets.WHAM,
                      tmpdir, sample_rate=44100)


def test_dataset_hook_slakh(benchmark_audio):
    # make a fake slakh directory.
    band = {"guitar": [30, 31], "drums": [127]}
    only_guitar = {"guitar": [30, 31]}
    empty = {}
    bad = {"guitar": [30], "guitar_2": [30]}
    with tempfile.TemporaryDirectory() as tmpdir:
        train_track_dir = os.path.join(tmpdir, "Track00001")
        # The test track doesn't contain any audio files.
        test_track_dir = os.path.join(tmpdir, "Track02000")
        os.mkdir(train_track_dir)
        os.mkdir(test_track_dir)
        # Create Metadata file
        metadata = yaml.dump({
          'audio_dir': 'stems',
          'midi_dir': 'MIDI',
          'stems': {
            'S00': {'program_num': 30},
            'S01': {'program_num': 127},
            'S02': {'program_num': 30},
            'S03': {'program_num': 30},
            'S04': {'program_num': 30},  # Not Synthesized
            'S05': {'program_num': 0}  # out of recipe
          }
        }, Dumper=Dumper)

        metadata_path = os.path.join(train_track_dir, "metadata.yaml")
        metadata_file = open(metadata_path, "w")
        metadata_file.write(metadata)
        metadata_file.close()

        metadata_path = os.path.join(test_track_dir, "metadata.yaml")
        metadata_file = open(metadata_path, "w")
        metadata_file.write(metadata)
        metadata_file.close()

        stems_dir = os.path.join(train_track_dir, "stems")
        midi_dir = os.path.join(train_track_dir, "MIDI")
        os.mkdir(stems_dir)
        os.mkdir(midi_dir)
        os.mkdir(os.path.join(test_track_dir, "stems"))
        test_midi_dir = os.path.join(test_track_dir, "MIDI")
        os.mkdir(test_midi_dir)

        # Note: These aren't actually guitar and drums
        guitar_path1 = os.path.join(stems_dir, "S00.wav")
        guitar_path2 = os.path.join(stems_dir, "S02.wav")
        guitar_path3 = os.path.join(stems_dir, "S03.wav")
        drums_path = os.path.join(stems_dir, "S01.wav")
        mix_path = os.path.join(train_track_dir, "mix.wav")

        # making midi objects
        midi_0 = PrettyMIDI()
        midi_1 = PrettyMIDI()
        guitar = Instrument(30, name="guitar")
        guitar.notes = [Note(70, 59, 0, 1)]
        drum = Instrument(127, is_drum=True, name="drum")
        drum.notes = [Note(40, 30, 0, 1)]
        midi_0.instruments.append(guitar)
        midi_1.instruments.append(drum)
        midi_1.write(os.path.join(midi_dir, "S01.mid"))
        midi_1.write(os.path.join(test_midi_dir, "S01.mid"))
        midi0_paths = ["S00.mid", "S02.mid", "S03.mid", "S04.mid"]
        for m in midi0_paths:
            midi_0.write(os.path.join(midi_dir, m))
            midi_0.write(os.path.join(test_midi_dir, m))

        midi_mix = PrettyMIDI()
        midi_mix.instruments += [guitar, drum]
        midi_mix.write(os.path.join(train_track_dir, "all_src.mid"))

        # Move them within directory
        shutil.copy(benchmark_audio['K0140.wav'], guitar_path1)
        shutil.copy(benchmark_audio['K0149.wav'], drums_path)
        # Make a mix from them.
        guitar_signal = nussl.AudioSignal(path_to_input_file=guitar_path1)
        drums_signal = nussl.AudioSignal(path_to_input_file=drums_path)
        guitar_signal.truncate_seconds(2)
        drums_signal.truncate_seconds(2)
        mix_signal = guitar_signal * 3 + drums_signal

        # Save audio objects
        mix_signal.write_audio_to_file(mix_path)
        drums_signal.write_audio_to_file(drums_path)
        guitar_signal.write_audio_to_file(guitar_path1)
        guitar_signal.write_audio_to_file(guitar_path3)
        guitar_signal.write_audio_to_file(guitar_path2)

        # now that our fake slakh has been created, lets try some mixing
        band_slakh = nussl.datasets.Slakh(
            tmpdir, recipe=band, midi=True, make_submix=True)
        # Our dataset should only have one item
        assert len(band_slakh) == 1
        data = band_slakh[0]
        _mix_signal, _sources = data["mix"], data["sources"]
        assert len(_sources) == 2

        # Checking audio
        assert np.allclose(mix_signal.audio_data, _mix_signal.audio_data)
        assert np.allclose(
            _sources["drums"].audio_data, drums_signal.audio_data)
        # Only three synthesized audio sources
        assert np.allclose(
            _sources["guitar"].audio_data, guitar_signal.audio_data * 3)
        _midi_mix, _midi_sources = data["midi_mix"], data["midi_sources"]

        # Checking midi
        # There are 4 guitar sources, but only 3 are synthesized
        assert len(_midi_mix.instruments) == 5
        assert len(_midi_sources) == 2
        assert _midi_sources["guitar"][0].instruments[0].program == 30
        assert _midi_sources["drums"][0].instruments[0].program == 127
        # Order should be in numeric order on the metadata.yaml file
        assert all(
            [instrument.program == program_num
             for instrument, program_num
             in zip(_midi_mix.instruments, [30, 127, 30, 30, 30])
             ]
        )

        # Checking non-submixing
        band_slakh = nussl.datasets.Slakh(
            tmpdir, recipe=band, midi=False, make_submix=False)
        data = band_slakh[0]
        _mix_signal, _sources = data["mix"], data["sources"]
        assert isinstance(_sources["guitar"], list)
        assert isinstance(_sources["drums"], list)
        assert len(_sources) == 2
        assert len(_sources["guitar"]) == 4
        assert len(_sources["drums"]) == 1
        assert np.allclose(_sources["guitar"]
                           [0].audio_data, guitar_signal.audio_data)
        assert np.allclose(
            sum(_sources["guitar"]).audio_data, 3 * guitar_signal.audio_data)
        # Check the last guitar source is empty
        audio_data = _sources["guitar"][-1].audio_data
        assert np.allclose(audio_data, np.zeros_like(audio_data))
        # Checking Lack of midi
        assert data.get("midi_mix", None) is None
        assert data.get("midi_sources", None) is None

        with pytest.raises(DataSetException):
            not_enough_instruments = nussl.datasets.Slakh(  # noqa
                tmpdir,
                recipe=band,
                midi=True,
                make_submix=True,
                min_acceptable_sources=3
            )
        # single source slakh
        guitar_slakh = nussl.datasets.Slakh(
            tmpdir, recipe=only_guitar, make_submix=True, min_acceptable_sources=1)
        data = guitar_slakh[0]
        _guitar_signal, _sources = data["mix"], data["sources"]
        assert len(_sources) == 1
        assert np.allclose(
            _sources["guitar"].audio_data, guitar_signal.audio_data * 3)
        assert np.allclose(_guitar_signal.audio_data,
                           guitar_signal.audio_data * 3)

        # Different split
        all_slakh = nussl.datasets.Slakh(
            tmpdir, recipe=band, split='all'
        )
        assert len(all_slakh) == 2

        # default_recipe
        default_slakh = nussl.datasets.Slakh(
            tmpdir, min_acceptable_sources=1
        )
        assert len(default_slakh.recipe.keys()) == 25

        # Error checking
        with pytest.raises(DataSetException):
            empty_slakh = nussl.datasets.Slakh(  # noqa
                tmpdir, recipe=empty, min_acceptable_sources=1)
        with pytest.raises(ValueError):
            nussl.datasets.Slakh(tmpdir, recipe=band, split='')
        with pytest.raises(ValueError):
            nussl.datasets.Slakh(tmpdir, recipe=band, min_acceptable_sources=0)
        with pytest.raises(ValueError):
            nussl.datasets.Slakh(tmpdir, recipe=bad)


def test_dataset_hook_fuss(scaper_folder):
    pytest.raises(DataSetException, nussl.datasets.FUSS, 'folder',
                  split='bad split')

    with tempfile.TemporaryDirectory() as tmpdir:
        train_folder = os.path.join(tmpdir, 'train')
        shutil.copytree(scaper_folder, train_folder)

        fuss = nussl.datasets.FUSS(tmpdir)


def test_dataset_hook_on_the_fly():
    def make_sine_wave(freq, sample_rate, duration):
        dt = 1 / sample_rate
        x = np.arange(0.0, duration, dt)
        x = np.sin(2 * np.pi * freq * x)
        return x

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
            source_data = make_sine_wave(freq, sample_rate, duration)
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
    assert len(dataset) == 10

    for output in dataset:
        assert output['mix'] == sum(output['sources'].values())
        assert output['mix'].signal_duration == duration

    def bad_mix_closure(dataset, i):
        return 'not a dictionary'
    pytest.raises(DataSetException, nussl.datasets.OnTheFly,
                  bad_mix_closure, 10)

    def bad_dict_closure(dataset, i):
        return {'key': 'no mix in this dict'}
    pytest.raises(DataSetException, nussl.datasets.OnTheFly,
                  bad_dict_closure, 10)

    def bad_dict_sources_closure(dataset, i):
        return {'mix': 'no sources in this dict'}
    pytest.raises(DataSetException, nussl.datasets.OnTheFly,
                  bad_dict_sources_closure, 10)
