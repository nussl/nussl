import librosa
import numpy as np
import sys
import nussl
print dir(nussl)

def load_audio(audio_file):
    print 'Loading %s' % audio_file
    audio, sr = librosa.load(audio_file, sr=None, mono=True, offset = 90, duration = 60)
    pad_length = (audio.shape[-1] / 1024.0 % 1) * 1024
    audio = np.pad(np.array(audio), ((0, int(pad_length))), 'constant', constant_values = (0, 0))
    audio_signal = nussl.AudioSignal(audio_data_array = audio, sample_rate = sr)
    audio_signal.stft_params.hop_length = 1024
    audio_signal.stft_params.n_fft_bins = 4096
    audio_signal.stft_params.window_length = 4096
    return audio_signal

mixture = load_audio('../Input/mixture.wav')
overlap_add = nussl.OverlapAdd(mixture, separation_method = 'REPET')
overlap_add.run()
bg, fg = overlap_add.make_audio_signals()
fg.write_audio_to_file('foreground_repet.wav')
bg.write_audio_to_file('background_repet.wav')

"""
overlap_add = nussl.OverlapAdd(mixture, separation_method = 'REPET-SIM')
overlap_add.run()
bg, fg = overlap_add.make_audio_signals()
fg.write_audio_to_file('foreground_repet_sim.wav')
bg.write_audio_to_file('background_repet_sim.wav')

repet_sim = nussl.RepetSim(mixture)
repet_sim.run()
bg, fg = repet_sim.make_audio_signals()
fg.write_audio_to_file('foreground_repet_sim_no_overlap.wav')
bg.write_audio_to_file('background_repet_sim_no_overlap.wav')

overlap_add = nussl.OverlapAdd(mixture, separation_method = 'FT2D')
overlap_add.run()
bg, fg = overlap_add.make_audio_signals()
fg.write_audio_to_file('foreground_ft2d.wav')
bg.write_audio_to_file('background_ft2d.wav')
"""

