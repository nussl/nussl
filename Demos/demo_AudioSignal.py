import nussl
import numpy as np
import scipy.io.wavfile as wav

def main():
    sr1, ch1 = wav.read('../Input/src1.wav')
    sr2, ch2 = wav.read('../Input/src2.wav')

    arr = np.array([ch1, ch2[0:sr1*10]], dtype=int)

    #sig2 = nussl.AudioSignal('../Input/src1.wav')
    signal = nussl.AudioSignal(audio_data_array=arr)

    n_ch = signal.num_channels
    signal.write_audio_to_file('../Output/duet_test1.wav')

if __name__ == '__main__':
    main()