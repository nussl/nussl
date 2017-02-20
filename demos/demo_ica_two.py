import os
import sys
import matplotlib.pyplot as plt

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)

from nussl import AudioSignal, ICA

observation = AudioSignal('../input/ica_demo.flac')

ica = ICA(input_audio_signal=observation)
ica.run()
sources = ica.make_audio_signals()
estimated = []
for i,s in enumerate(sources):
    s.write_audio_to_file('output/ICA %d.wav' % i)