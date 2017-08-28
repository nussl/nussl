#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple example of NMF in nussl showing some common-used features.
"""
import os

import numpy as np
import nussl


def simple_nmf():
    """
    A simple example of NMF in nussl
    """
    # Make a simple matrix
    n = 50
    mixture = np.ones((n, n))

    num_lines = 12
    val = 10

    for i in np.arange(0, n+1, n / num_lines):
        mixture[i, :] += val

    # Make a cool design
    mixture[n/3:, :n/3] = 1
    mixture[2*n/3:, n/3:2*n/3] = 1
    mixture[:2*n/3, 2*n/3:] = 1
    mixture[:n/3, n/3:] = 1

    # Set up NMF
    n_iter = 10
    nmf = nussl.TransformerNMF(mixture, num_components=num_lines*3, max_num_iterations=n_iter,
                               distance_measure=nussl.TransformerNMF.KL_DIVERGENCE, seed=0)

    # run  NMF
    nmf.transform()

    output_path = os.path.abspath(os.path.join('Output', 'simple_nmf.png'))
    nmf.plot(output_file=output_path, matrix_to_db=False)


if __name__ == '__main__':
    simple_nmf()
