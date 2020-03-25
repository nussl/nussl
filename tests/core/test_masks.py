import nussl
import pytest
import numpy as np
from nussl.core.audio_signal import AudioSignalException
from nussl.core.masks import BinaryMask, SoftMask, MaskBase
from copy import deepcopy

sr = nussl.constants.DEFAULT_SAMPLE_RATE
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


def test_create_mask():
    mask_data = np.random.rand(1025, 400, 1)
    pytest.raises(NotImplementedError, lambda x: MaskBase(x), mask_data)
    pytest.raises(ValueError, lambda x: MaskBase(), mask_data)
    pytest.raises(NotImplementedError,
                  lambda x: MaskBase._validate_mask(mask_data), mask_data)
    pytest.raises(ValueError, lambda x: MaskBase(
        input_mask=mask_data, mask_shape=mask_data.shape), mask_data)
    bad_data = np.random.rand(1025, 400, 1, 1)
    pytest.raises(ValueError, lambda x: SoftMask(input_mask=x), bad_data)
    bad_data = np.random.rand(1025)
    pytest.raises(ValueError, lambda x: SoftMask(input_mask=x), bad_data)
    pytest.raises(ValueError, SoftMask, [0])

    pytest.raises(ValueError, lambda x: SoftMask(input_mask=x), mask_data > .5)

    s1 = SoftMask(mask_data)
    thresholds = [.1, .2, .5, 1.0]
    for t in thresholds:
        binary_mask = s1.mask_to_binary(t)
        assert np.allclose(binary_mask.mask, mask_data > t)

    m1 = np.zeros((1025, 400, 1))
    s1 = SoftMask.zeros(m1.shape)
    assert np.allclose(s1.mask, m1)

    m1 = np.ones((1025, 400, 1))
    s1 = SoftMask.ones(m1.shape)
    assert np.allclose(s1.mask, m1)

    m1 = np.ones((1025, 400, 1)).astype(float)
    s1 = SoftMask(m1)
    assert s1.dtype == float
    assert s1.shape == (1025, 400, 1)
    assert s1.num_channels == 1

    m1 = np.zeros((1025, 400, 1))
    s1 = SoftMask(mask_shape=m1.shape)
    assert np.allclose(s1.mask, m1)


def test_binary_mask():
    mask_data = np.random.rand(1025, 400, 2)
    mask_data = mask_data > .5
    binary_mask = BinaryMask(mask_data)

    assert np.allclose(binary_mask.mask_as_ints(), mask_data.astype(int))

    for ch in range(mask_data.shape[-1]):
        int_mask = binary_mask.mask_as_ints(ch)
        assert np.allclose(int_mask, mask_data[..., ch].astype(int))

    pytest.raises(ValueError, BinaryMask, mask_data + 1)
    pytest.raises(ValueError, BinaryMask, mask_data * .5)

    binary_mask = BinaryMask(mask_data * .9999)
    assert np.allclose(binary_mask.mask, mask_data)


def test_mask_get_channels():
    mask_data = np.random.rand(1025, 400, 2)
    soft_mask = SoftMask(mask_data)
    for ch in range(mask_data.shape[-1]):
        _mask_ch = soft_mask.get_channel(ch)
        assert np.allclose(_mask_ch, mask_data[..., ch])

    pytest.raises(ValueError, soft_mask.get_channel, 2)
    pytest.raises(ValueError, soft_mask.get_channel, -1)


def test_mask_arithmetic():
    m1 = np.random.rand(1025, 400, 1)
    m2 = np.random.rand(1025, 400, 1)
    s1 = SoftMask(m1)
    s2 = SoftMask(m2)

    pytest.raises(ValueError, lambda x: x * [0], s1)
    pytest.raises(ValueError, lambda x: x + [0], s1)

    r1 = s1 + s2
    assert np.allclose(r1.mask, m1 + m2)

    r1 = s1 + s2.mask
    assert np.allclose(r1.mask, m1 + m2)

    r1 = s1 - s2
    assert np.allclose(r1.mask, m1 - m2)

    r1 = s1 * 2
    assert np.allclose(r1.mask, m1 * 2)

    r1 = 2 * s1
    assert np.allclose(r1.mask, m1 * 2)

    r1 = deepcopy(s1)
    r1 += s1
    assert np.allclose(r1.mask, m1 * 2)

    r1 = deepcopy(s1)
    r1 -= s1
    assert np.allclose(r1.mask, m1 * 0)

    r1 = deepcopy(s1)
    r1 *= 2
    assert np.allclose(r1.mask, m1 * 2)

    r1 = deepcopy(s1)
    r1 /= .5
    assert np.allclose(r1.mask, m1 * 2)

    r2 = SoftMask(m1 * 2)
    assert r1 == r2
    r2 *= .5
    assert r1 != r2


def test_masks_sum_to_mix(benchmark_audio):
    for key, path in benchmark_audio.items():
        signal = nussl.AudioSignal(path)
        signal.stft()

        for num_sources in range(1, 4):
            shape = signal.stft_data.shape + (num_sources,)

            random_masks = np.random.random(shape)
            random_masks = (
                    random_masks /
                    np.sum(random_masks, axis=-1, keepdims=True)
            )
            estimates = []

            for i in range(num_sources):
                mask = SoftMask(random_masks[..., i])
                estimate = signal.apply_mask(mask)
                estimate.istft()
                estimates.append(estimate)

            assert np.allclose(
                sum(estimates).audio_data,
                signal.audio_data,
                atol=stft_tol
            )
