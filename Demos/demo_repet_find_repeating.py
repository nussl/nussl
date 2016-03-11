import nussl

def main():
    a = nussl.AudioSignal('../Input/saw1_noise_time_0.05.wav')
    r = nussl.Repet(a)

    beat_spectrum = r.get_beat_spectrum()

    r.update_periods()
    repeating_period = r.find_repeating_period_simple(beat_spectrum, r.min_period, r.max_period)

    # repeating_period based on the stft so is a multiple of hop, so have to convert it
    repeating_period_in_seconds = r.stft_params.hop_length * repeating_period / a.sample_rate

    print repeating_period, repeating_period_in_seconds

if __name__ == '__main__':
    main()