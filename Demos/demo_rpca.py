import librosa
import numpy as np
import sys
import nussl

def load_audio(audio_file):
    print 'Loading %s' % audio_file
    audio, sr = librosa.load(audio_file, sr=None, mono=True)
    pad_length = (audio.shape[-1] / 1024.0 % 1) * 1024
    #audio = np.pad(np.array(audio), ((0, int(pad_length))), 'constant', constant_values = (0, 0))
    audio_signal = nussl.AudioSignal(audio_data_array = audio, sample_rate = sr)
    audio_signal.stft_params.hop_length = 256
    audio_signal.stft_params.window_length = 1024
    return audio_signal

mixture = load_audio('./Input/titon_2_07_SNR5.wav')
rpca = nussl.RPCA(mixture)
rpca.run()
bg, fg = rpca.make_audio_signals()
fg.write_audio_to_file('foreground_rpca_titon.wav')
bg.write_audio_to_file('background_rpca_titon.wav')
