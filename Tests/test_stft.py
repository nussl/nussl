import librosa
import os
import sys
import numpy as np
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)

import nussl
print nussl.USE_LIBROSA_STFT

y, sr = librosa.load(librosa.util.example_audio_file())

def test_stft_istft(y, sr, window_length, hop_length, window_type):
    signal = nussl.AudioSignal(audio_data_array = y, sample_rate = sr)
    print signal.signal_length
    signal.stft_params.window_length = window_length
    signal.stft_params.n_fft_bins = window_length
    signal.stft_params.hop_length = hop_length
    signal.stft()
    signal.istft()
    print signal.signal_length

    print 'nussl reconstruction error'
    print 'window_length: %d, hop_length: %d' % (window_length, hop_length)
    print np.sum(np.abs(y - signal.audio_data))

test_stft_istft(y, sr, 4096, 1024, 'hamming')
test_stft_istft(y, sr, 2048, 1024, 'hamming')
test_stft_istft(y, sr, 4096, 512, 'hamming')
test_stft_istft(y, sr, 2048, 1536, 'hamming')
