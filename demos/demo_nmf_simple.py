#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo for NMF as a general matrix decomposition tool in nussl
"""

from __future__ import print_function
import time
import os
import sys
import numpy as np


try:
    # import from an already installed version
    import nussl
except:

    # can't find an installed version, import from right next door...
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not path in sys.path:
        sys.path.insert(1, path)

    import nussl


def simple_example():
    """
    A simple example of using NMF as a matrix decomposer (transformer) in nussl
    """
    # Make two simple matrices
    n = 4
    a = np.arange(n ** 2).reshape((n, n))
    b = np.add(np.multiply(2, a), 3)

    # Mix them together
    mixture = np.dot(b, a)

    # Set up TransformerNMF
    num_templates = 2
    nmf = nussl.transformers.TransformerNMF(mixture, num_components=num_templates, max_num_iterations=100,
                                            distance_measure=nussl.transformers.TransformerNMF.EUCLIDEAN)

    # run NMF
    start = time.time()
    nmf.transform()
    print('{0:.3f} seconds for NUSSL'.format(time.time() - start))

    print('Original mixture =\n {}'.format(mixture))
    print('NMF Reconstructed mixture =\n {}'.format(np.dot(nmf.template_dictionary, nmf.activation_matrix)))


if __name__ == '__main__':
    simple_example()
