#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import numpy as np
import nussl

def simple_example():
    """
    A simple example of using NMF as a transformer in nussl
    """
    # Make two simple matrices
    n = 4
    a = np.arange(n ** 2).reshape((n, n))
    b = np.add(np.multiply(2, a), 3)

    # Mix them together
    mixture = np.dot(b, a)

    # Set up NU NMF
    num_templates = 2
    nmf = nussl.transformers.TransformerNMF(mixture, num_templates=num_templates)
    nmf.should_do_epsilon = False
    nmf.max_num_iterations = 100
    nmf.distance_measure = nmf.EUCLIDEAN

    # run NMF
    start = time.time()
    nmf.transform()
    print('{0:.3f} seconds for NUSSL'.format(time.time() - start))

    print('Original mixture =\n {}'.format(mixture))
    print('NMF Reconstructed mixture =\n {}'.format(np.dot(nmf.templates, nmf.activation_matrix)))


if __name__ == '__main__':
    simple_example()
