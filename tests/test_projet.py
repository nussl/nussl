import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)

import nussl
print nussl

mixture = nussl.AudioSignal('../input/panned_mix.wav')

projet = nussl.Projet(mixture, verbose = True, num_iterations = 50, num_sources = 2)
projet.run()
sources = projet.make_audio_signals()

for i,m in enumerate(sources):
    m.write_audio_to_file('../input/projet_%d.wav' % i)
