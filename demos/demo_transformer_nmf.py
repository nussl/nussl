import time
import numpy as np
import nussl
import nimfa

def simple_example():
    """
    A simple example of NMF in nussl
    """
    # Make two simple matrices
    n = 4
    a = np.arange(n ** 2).reshape((n, n))
    b = 2. * a + 3.

    # Mix them together
    mixture = np.dot(b, a)

    # Set up NU NMF
    n_bases = 2
    n_iter = 100
    nussl_nmf = nussl.TransformerNMF(mixture, n_bases)
    nussl_nmf.should_use_epsilon = False
    nussl_nmf.max_num_iterations = n_iter
    nussl_nmf.distance_measure = nussl.DistanceType.DIVERGENCE

    # run NU NMF
    nussl_nmf.run()
    signals = nussl_nmf.recombine_calculated_matrices()

if __name__ == '__main__':
    simple_example()