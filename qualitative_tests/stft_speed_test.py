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
    dur = np.arange(60, 311, 30)  # five second intervals from 5-120 seconds
    lengths = dur * sr
    trials = 25

    librosa_stft_times = {}
    librosa_istft_times = {}
    nussl_stft_times = {}
    nussl_istft_times = {}

    # Make the signals and test
    for cur_len in lengths:
        print('{}\n{}cur_len: {}\n{}\n'.format('*' * 50, ' ' * 20, cur_len / sr, '*' * 50))

        librosa_stft_times[cur_len] = []
        nussl_stft_times[cur_len] = []
        librosa_istft_times[cur_len] = []
        nussl_istft_times[cur_len] = []

        # sig = make_fm_signal(cur_len, sr)
        sig = np.array(np.sin(np.linspace(0, freq * 2 * np.pi, cur_len)))

        for i in range(trials):
            # print('{}{}\n{}trial: {}\n{}\n'.format(' ' * 20, '*' * 15, ' ' * 20, i, '*' * 15))
            # Make a new signal

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


def make_fm_signal(dur, sample_rate):
    dt = 1.0 / float(sample_rate)
    freq = 5000 # Hz
    x = np.arange(0.0, dur, dt)
    x = np.sin(2 * np.pi * freq * x)

    lfo_freq = 3
    lfo_amp = 800
    freq2 = 15000
    x2 = np.arange(0.0, dur, dt)
    modulator = lfo_amp * np.sin(2 * np.pi * lfo_freq * x2)
    x2 = np.sin(2 * np.pi * freq2 * x2 + modulator)
    x2 += x

    return x2

if __name__ == '__main__':
    speed_test()
