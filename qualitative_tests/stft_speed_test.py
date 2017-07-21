#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)

import nussl


def speed_test():
    freq = 3000
    sr = nussl.constants.DEFAULT_SAMPLE_RATE  # 44.1kHz
    dur = np.arange(5, 121, 5)  # five second intervals from 5-120 seconds
    lengths = dur * sr
    trials = 25

    librosa_stft_times = {}
    librosa_istft_times = {}
    nussl_stft_times = {}
    nussl_istft_times = {}

    # Make the signals and test
    for cur_len in lengths:
        librosa_stft_times[cur_len] = []
        nussl_stft_times[cur_len] = []
        librosa_istft_times[cur_len] = []
        nussl_istft_times[cur_len] = []

        for i in range(trials):
            # Make a new signal
            sig = np.array(np.sin(np.linspace(0, freq * 2 * np.pi, cur_len)))
            a = nussl.AudioSignal(audio_data_array=sig)

            n_st = time.time()
            nussl_stft = a.stft(use_librosa=False)
            n_end = time.time() - n_st
            nussl_stft_times[cur_len].append(n_end)

            a = nussl.AudioSignal(audio_data_array=sig)
            l_st = time.time()
            librosa_stft = a.stft(use_librosa=True)
            l_end = time.time() - l_st
            librosa_stft_times[cur_len].append(l_end)

            a = nussl.AudioSignal(stft=nussl_stft)
            n_st = time.time()
            a.istft(use_librosa=False)
            n_end = time.time() - n_st
            nussl_istft_times[cur_len].append(n_end)

            a = nussl.AudioSignal(stft=librosa_stft)
            l_st = time.time()
            a.istft(use_librosa=True)
            l_end = time.time() - l_st
            librosa_istft_times[cur_len].append(l_end)

    # Average over all trials
    nussl_stft_avg = []
    librosa_stft_avg = []
    nussl_istft_avg = []
    librosa_istft_avg = []
    for cur_len in lengths:
        nussl_stft_avg.append(np.mean(nussl_stft_times[cur_len]))
        librosa_stft_avg.append(np.mean(librosa_stft_times[cur_len]))
        nussl_istft_avg.append(np.mean(nussl_istft_times[cur_len]))
        librosa_istft_avg.append(np.mean(librosa_istft_times[cur_len]))

    # Plot STFT avgs
    plt.plot(lengths / sr, nussl_stft_avg, label='nussl stft')
    plt.plot(lengths / sr, librosa_stft_avg, label='librosa stft')
    plt.xlabel('Length of signal (sec)')
    plt.ylabel('Time to do STFT (sec)')
    plt.legend(loc='lower right')
    plt.title('Average time taken to do STFT over {} trials'.format(trials))
    plt.savefig('speed_test_stft.png')

    # Plot iSTFT avgs
    plt.close('all')
    plt.plot(lengths / sr, nussl_istft_avg, label='nussl istft')
    plt.plot(lengths / sr, librosa_istft_avg, label='librosa istft')
    plt.xlabel('Length of signal (sec)')
    plt.ylabel('Time to do iSTFT (sec)')
    plt.legend(loc='lower right')
    plt.title('Average time taken to do iSTFT over {} trials'.format(trials))
    plt.savefig('speed_test_istft.png')

if __name__ == '__main__':
    speed_test()
