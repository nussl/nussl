import numpy as np
from nussl.core.augmentation import augment
import nussl.datasets.hooks as hooks
import tempfile
import pytest
import librosa

# We test on a single item from MUSDB
musdb = hooks.MUSDB18(download=True)
item = musdb[40]

# Put sources in tempfiles
item_tempfile = tempfile.NamedTemporaryFile()
mix_loc = item_tempfile.name
item["mix"].write_audio_to_file(item_tempfile.name)
source_tempfiles = {}
for name, source in item["sources"].items():
    source_tempfile = tempfile.NamedTemporaryFile()
    source.write_audio_to_file(source_tempfile.name)
    source_tempfiles[name] = source_tempfile

dataset = [item]

@pytest.mark.xfail
def test_augment_params1():
    augment(dataset, augment_proportion=2)

@pytest.mark.xfail
def test_augment_params2():
    augment(dataset, num_augments=-1)

def test_stretch():
    stretch_factor = .8

    augmented_dataset = augment(dataset, time_stretch=(stretch_factor, stretch_factor))
    aug_item = augmented_dataset[0]
    assert np.allclose(aug_item["mix"].audio_data[1, :], 
        librosa.effects.time_stretch(np.asfortranarray(item["mix"].audio_data[1, :]), stretch_factor))
    for name, source in item["sources"].items():
        assert np.allclose(aug_item["sources"][name].audio_data[1, :], 
            librosa.effects.time_stretch(np.asfortranarray(item["sources"][name].audio_data[1, :]), stretch_factor))

def test_pitch_shift():
    shift = 2
    sample_rate = item["mix"].sample_rate

    augmented_dataset = augment(dataset, pitch_shift=(shift, shift))
    aug_item = augmented_dataset[0]
    assert np.allclose(aug_item["mix"].audio_data[1, :], 
        librosa.effects.pitch_shift(np.asfortranarray(item["mix"].audio_data[1, :]), sample_rate, shift))
    for name, source in item["sources"].items():
        assert np.allclose(aug_item["sources"][name].audio_data[1, :], 
            librosa.effects.pitch_shift(np.asfortranarray(item["sources"][name].audio_data[1, :]), sample_rate, shift))


