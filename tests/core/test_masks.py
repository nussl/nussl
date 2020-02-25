import nussl
import pytest
import numpy as np
import tempfile
import librosa
from nussl.core.audio_signal import AudioSignalException

sr = nussl.DEFAULT_SAMPLE_RATE
dur = 3  # seconds
length = dur * sr

def test_apply_mask(benchmark_audio):
    for key, path in benchmark_audio.items():
        signal = nussl.AudioSignal(path)