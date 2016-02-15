import unittest
import numpy as np
import nussl


class TestFftUtils(unittest.TestCase):
    sr = 44100
    dur = 3
    length = sr * dur
    n_ch = 1

    win_min = 7   # 2 ** 7  =  128
    win_max = 13  # 2 ** 13 = 8192
    win_lengths = [2 ** i for i in range(win_min, win_max)]

    hop_length_ratios = [1.0, 0.75, 0.5, 0.25]

    win_length_40ms = int(2 ** (np.ceil(np.log2(nussl.Constants.DEFAULT_WIN_LEN_PARAM * sr))))

    def test_stft_istft_noise_seed1(self):
        win_type = nussl.WindowType.RECTANGULAR

        for win_length in self.win_lengths:
            for i in range(10):
                hop_length = win_length

                np.random.seed(i)
                noise = (np.random.rand(self.n_ch, self.length) * 2) - 1
                noise = noise[0, ]

                self.do_stft_istft_assert_allclose(win_length, hop_length, win_type, noise)

    def test_stft_istft_noise_no_seed(self):
        win_type = nussl.WindowType.RECTANGULAR

        for win_length in self.win_lengths:
            hop_length = win_length
            noise = (np.random.rand(self.n_ch, self.length) * 2) - 1
            noise = noise[0, ]

            self.do_stft_istft_assert_allclose(win_length, hop_length, win_type, noise)

    def test_stft_istft_ones1(self):
        '''
        Tests all ones with rectangular window at a bunch of different lengths
        hop length is same as window length
        '''
        win_type = nussl.WindowType.RECTANGULAR
        ones = np.ones(self.length)

        for win_length in self.win_lengths:
            hop_length = win_length

            self.do_stft_istft_assert_allclose(win_length, hop_length, win_type, ones)

    def test_stft_istft_ones2(self):

        win_type = nussl.WindowType.RECTANGULAR
        ones = np.ones(self.length)

        for win_length in self.win_lengths:
            for i in self.hop_length_ratios:
                hop_length = int(win_length * i)

                self.do_stft_istft(win_length, hop_length, win_type, ones)

    def test_stft_istft_hann1(self):
        win_type = nussl.WindowType.HANN

        for win_length in self.win_lengths:
            hop_length = win_length / 2
            ones = np.ones(self.length)

            self.do_stft_istft_assert_allclose(win_length, hop_length, win_type, ones)

    def test_stft_istft_40ms_win_length(self):
        win_type = nussl.WindowType.RECTANGULAR
        ones = np.ones(self.length)


        for i in self.hop_length_ratios:
            hop_legnth = int(self.win_length_40ms * i)

            self.do_stft_istft(self.win_length_40ms, hop_legnth, win_type, ones)

    def test_e_stft_plus(self):
        """
        This test just has to not crash!
        :return:
        """
        # win_type = nussl.WindowType.RECTANGULAR
        win_type = 'rectangular'
        ones = np.ones(self.length)

        stft, psd, f, t = nussl.e_stft_plus(ones, self.win_length_40ms, self.win_length_40ms, win_type, self.sr)

    @staticmethod
    def do_stft_istft_assert_allclose(win_length, hop_length, win_type, signal):
        signal_1, calculated_signal = TestFftUtils.do_stft_istft(win_length, hop_length, win_type, signal)

        # leave off comparing the first and last hop to mitigate edge effects
        assert (np.allclose(signal_1, calculated_signal))


    @staticmethod
    def do_stft_istft(win_length, hop_length, win_type, signal):
        stft = nussl.e_stft(signal, win_length, hop_length, win_type)

        # get rid of last hop because it's zero padded and screws up the stft and np.allclose
        length = int(len(signal) / hop_length) * hop_length
        calculated_signal = nussl.e_istft(stft, win_length, hop_length, win_type)

        # useful for debugging:
        # diff = signal[hop_length:length] - calculated_signal[hop_length:length]

        return signal[hop_length:length], calculated_signal[hop_length:length]


if __name__ == '__main__':
    unittest.main()