import numpy as np
import numpy.random as random
from nussl.core.augmentation import *
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


def test_stretch():
    stretch_factor = .8

    augmented = time_stretch(item["mix"], stretch_factor)

    assert np.allclose(augmented.audio_data[1, :], 
            librosa.effects.time_stretch(np.asfortranarray(item["mix"].audio_data[1, :]), stretch_factor))

def test_pitch_shift():
    shift = 2
    sample_rate = item["mix"].sample_rate

    augmented = pitch_shift(item["mix"], shift)

    assert np.allclose(augmented.audio_data[1, :], 
            librosa.effects.pitch_shift(np.asfortranarray(item["mix"].audio_data[1, :]), sample_rate, shift))

