#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Second demo of ICA in nussl
"""

import os
import sys

try:
    # import from an already installed version
    import nussl
except:

    # can't find an installed version, import from right next door...
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not path in sys.path:
        sys.path.insert(1, path)

    import nussl


def main():
    """
    Simple example of ICA with an actual audio file in nussl
    Returns:

    """
    observation = nussl.AudioSignal(os.path.join('..', 'input', 'ica_demo.flac'))

    ica = nussl.separation.ICA(observations_list=observation)
    ica.run()

    sources = ica.make_audio_signals()
    for i,s in enumerate(sources):
        s.write_audio_to_file('output/ICA_{}.wav'.format(i))


if __name__ == '__main__':
    main()