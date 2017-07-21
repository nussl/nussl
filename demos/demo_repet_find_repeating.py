import nussl

def main():
    a = nussl.AudioSignal('../Input/HistoryRepeating_PropellorHeads.wav')
    r = nussl.Repet(a)

    beat_spectrum = r.get_beat_spectrum()

    r.update_periods()
    period_simple = r.find_repeating_period_simple(beat_spectrum, r.min_period, r.max_period)
    period_complex = float(r.find_repeating_period_complex(beat_spectrum))

    # repeating_period based on the stft so is a multiple of hop, so have to convert it
    period_simple_seconds = r.stft_params.hop_length * period_simple / a.sample_rate
    period_complex_seconds = r.stft_params.hop_length * period_complex / a.sample_rate

    print 'simple = ', period_simple,'hops,',  period_simple_seconds, 'seconds'
    print 'complex = ', period_complex,'hops,',  period_complex_seconds, 'seconds'

if __name__ == '__main__':
    main()