import nussl
import librosa
import os
import matplotlib
import time
import numpy as np
from mir_eval.separation import bss_eval_sources

def load_audio(audio_file):
    print 'Loading %s' % audio_file
    audio, sr = librosa.load(audio_file, sr=None, mono=True, offset = 60, duration = 60)
    pad_length = (audio.shape[-1] / 1024.0 % 1) * 1024
    audio = np.pad(np.array(audio), ((0, int(pad_length))), 'constant', constant_values = (0, 0))
    audio_signal = nussl.AudioSignal(audio_data_array = audio, sample_rate = sr)
    return audio_signal

def repet_sim(audio_signal):
    repet = nussl.RepetSim(audio_signal)
    repet.stft_params.hop_length = 1024
    repet.stft_params.n_fft_bins = 4096
    repet.stft_params.window_length = 4096
    repet.run()
    bg, fg = repet.make_audio_signals()
    return bg, fg

def separate(audio_file, method):
    audio_signal = load_audio(audio_file)
    start_time = time.time()
    if method == 'repet_sim':
        bg, fg = repet_sim(audio_signal)
    end_time = time.time()
    return bg, fg, end_time - start_time

methods = ['repet_sim']

def evaluate(reference, estimated):
        estimated = np.vstack([x.audio_data[0, :] for x in estimated])
        reference = np.vstack([x.audio_data[0, :] for x in reference])
        print estimated.shape[-1], reference.shape[-1]
        print estimated.shape[-1] - reference.shape[-1]
        L = min(estimated.shape[-1], reference.shape[-1])
        estimated = estimated[:, 0:L]
        reference = reference[:, 0:L]
        return bss_eval_sources(reference, estimated, compute_permutation = False)
    
def separate_and_evaluate(audio_file, voice_file):
    mixture = load_audio(audio_file)
    foreground = load_audio(voice_file)
    background = mixture - foreground

    sr = mixture.sample_rate
    sources = {method: [] for method in methods}
    result = {}
    for method in methods:
        print 'Separating'
        bg, fg, time_taken = separate(audio_file, method)
        sources[method].append((bg, fg, time_taken, audio_file))
        print 'Evaluating'
        result[method] = evaluate([background, foreground], sources[method][0][0:2])
        print 'Done evaluating'
    return result

separate_and_evaluate('mixture.wav', 'vocals.wav')
