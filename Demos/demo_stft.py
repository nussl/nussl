import nussl
import numpy as np

def main():
    signal = nussl.AudioSignal('../Input/mix1.wav')

    # signal.stft_params.window_length = int(0.04* signal.sample_rate)
    # signal.stft_params.num_fft_bins = signal.stft_params.window_length
    # signal.stft_params.window_overlap_ratio = 0.5

    # signal.stft()
    # signal.istft()
    # signal.plot_spectrogram('../Output/police_spect.png')

    nussl.plot_stft(signal.get_channel(1), '../Output/mix1_1.png')


if __name__ == '__main__':
    main()