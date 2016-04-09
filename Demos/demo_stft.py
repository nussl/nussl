import nussl
import numpy as np

def main():
    win_type = nussl.WindowType.HANN
    sample_rate = nussl.DEFAULT_SAMPLE_RATE

    # Plot a simple sine wave at 5kHz
    dt = 1.0 / float(sample_rate)
    dur = 10.0 # sec
    freq = 5000 # Hz
    x = np.arange(0.0, dur, dt)
    x = np.sin(2 * np.pi * freq * x)

    signal = nussl.AudioSignal(audio_data_array=x)
    nussl.plot_stft(signal.get_channel(1), '../Output/sine_wave5000Hz.png', window_type=win_type)

    # Make a FM signal and plot it.
    lfo_freq = 3
    lfo_amp = 800
    freq2 = 15000
    x2 = np.arange(0.0, dur, dt)
    modulator = lfo_amp * np.sin(2 * np.pi * lfo_freq * x2)
    x2 = np.sin(2 * np.pi * freq2 * x2 + modulator)
    x2 += x

    signal2 = nussl.AudioSignal(audio_data_array=x2)
    nussl.plot_stft(signal2.get_channel(1), '../Output/fm_wave.png', window_type=win_type)

    # Plot from a file
    path = '../Input/police_noisy.wav'
    a = nussl.AudioSignal(path)
    nussl.plot_stft(a.get_channel(1), '../Output/police_noisy.png', freq_max=18000)

    # Plot through audio_signal
    a.plot_spectrogram()

    # plot glenn gould
    # glenn = nussl.AudioSignal('../Input/wtc-gould.mp3')
    # glenn.plot_spectrogram()


if __name__ == '__main__':
    main()