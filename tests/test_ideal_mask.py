#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)

import nussl

mixture = nussl.AudioSignal('../input/mixture/mixture.wav', duration = 30, offset = 60)
vocals = nussl.AudioSignal('../input/mixture/vocals.wav', duration = 30, offset = 60)
drums= nussl.AudioSignal('../input/mixture/drums.wav', duration = 30, offset = 60)

ideal_mask = nussl.IdealMask(mixture, sources = [vocals, drums])
ideal_mask.run()
masked_sources = ideal_mask.make_audio_signals()

for i,m in enumerate(masked_sources):
    m.write_audio_to_file('../input/mixture/ideal_mask_%d.wav' % i)