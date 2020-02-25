import nussl
import pytest
import numpy as np
import tempfile
import librosa
from nussl.core.audio_signal import AudioSignalException
from nussl.core.masks import BinaryMask, SoftMask

sr = nussl.DEFAULT_SAMPLE_RATE
dur = 3  # seconds
length = dur * sr
stft_tol = 1e-6

def test_apply_mask(benchmark_audio):
    for key, path in benchmark_audio.items():
        signal = nussl.AudioSignal(path)
        signal.stft()

        mask_data = np.random.rand(*signal.stft_data.shape)
        soft_mask = SoftMask(mask_data)
        inverse_mask = soft_mask.inverse_mask()
        bad_mask = SoftMask(mask_data[0])

        signal.stft_data = None
        pytest.raises(AudioSignalException, signal.apply_mask, soft_mask)
        signal.stft()
        pytest.raises(AudioSignalException, signal.apply_mask, [0])
        pytest.raises(AudioSignalException, signal.apply_mask, bad_mask)


        s1 = signal.apply_mask(soft_mask)
        s1.istft(truncate_to_length=signal.signal_length)
        s2 = signal.apply_mask(inverse_mask)
        s2.istft(truncate_to_length=signal.signal_length)

        recon = s1 + s2

        assert np.allclose(recon.audio_data, signal.audio_data, atol=stft_tol)

        mask_data = np.random.rand(*signal.stft_data.shape) > .5
        binary_mask = BinaryMask(mask_data)
        inverse_mask = binary_mask.inverse_mask()

        s1 = signal.apply_mask(binary_mask)
        s1.istft(truncate_to_length=signal.signal_length)
        s2 = signal.apply_mask(inverse_mask)
        s2.istft(truncate_to_length=signal.signal_length)

        recon = s1 + s2

        assert np.allclose(recon.audio_data, signal.audio_data, atol=stft_tol)

        signal.apply_mask(binary_mask, overwrite=True)

        assert np.allclose(signal.stft_data, s1.stft_data)