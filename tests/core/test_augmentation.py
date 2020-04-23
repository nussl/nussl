import numpy as np
from nussl.core.augmentation import augment
import tempfile
import pytest
import librosa

# We test on a single item from MUSDB
musdb = hooks.MUSDB18(download=True)
item = musdb[40]

# Put sources in tempfiles
item_tempfile = tempfile.NamedTemporaryFile()
mix_loc = item_tempfile.name
musdb["mix"].write_audio_to_file(item_tempfile.name)
source_tempfiles = {}
for name, source in musdb["sources"].items():
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
    augmented_dataset = augment(dataset, time_stretch=(.8, .8))
    aug_item = augmented_dataset[0]
    assert np.close(aug_item["mix"].audio_data, 
        librosa.time_stretch(item.audio_data, .8))
    for name, source in item["sources"].items():
        assert np.close(aug_item["sources"][name].item_audio_data, 
            librosa.time_stretch(source.audio_data, .8))

