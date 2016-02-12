import unittest
import numpy as np
import nussl


class TestFftUtils(unittest.TestCase):
    sr = 44100
    dur = 3
    length = sr * dur
    n_ch = 1
    win_min = 7 # 2 ** 7 = 128
    win_max = 13 # 2 ** 13 = 8192
    win_lengths = [2 ** i for i in range(win_min, win_max)]

    def test_stft_istft_noise_seed1(self):
        win_type = nussl.WindowType.RECTANGULAR

        for win_length in self.win_lengths:
            for i in range(10):
                hop_length = win_length

                np.random.seed(i)
                noise = (np.random.rand(self.n_ch, self.length) * 2) - 1
                noise = noise[0,]

                self.do_stft_istft(win_length, hop_length, win_type, noise)

    def test_stft_istft_noise_no_seed(self):
        win_type = nussl.WindowType.RECTANGULAR

        for win_length in self.win_lengths:
            hop_length = win_length
            noise = (np.random.rand(self.n_ch, self.length) * 2) - 1
            noise = noise[0,]

            self.do_stft_istft(win_length, hop_length, win_type, noise)

    def test_stft_istft_ones1(self):
        win_type = nussl.WindowType.RECTANGULAR

        for win_length in self.win_lengths:
            hop_length = win_length
            ones = np.ones(self.length)

            self.do_stft_istft(win_length, hop_length, win_type, ones)

    @staticmethod
    def do_stft_istft(win_length, hop_length, win_type, signal):
        stft = nussl.e_stft(signal, win_length, hop_length, win_type)

        # get rid of last hop because it's zero padded and screws up the stft and np.allclose
        length = int(len(signal) / hop_length) * hop_length
        calculated_signal = nussl.e_istft(stft, hop_length)

        # useful for debugging:
        # diff = signal[0:length] - calculated_signal

        assert (np.allclose(signal[0:length], calculated_signal[0:length]))


if __name__ == '__main__':
    unittest.main()