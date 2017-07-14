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
    b = 2 * a + 3

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
    # for sig in signals:
    #     print sig

    V = mixture
    nimfa_nmf = nimfa.Nmf(V, max_iter=n_iter, rank=n_bases, update='divergence')
    nmf_fit = nimfa_nmf()
    W = nmf_fit.basis()
    H = nmf_fit.coef()

    nimfa_est = np.dot(W, H)
    nussl_est = np.dot(nussl_nmf.templates, nussl_nmf.activation_matrix)
    error = np.abs(nussl_est - nimfa_est) / nimfa_est * 100

    print('nimfa estimate:\n%s\n\n' % nimfa_est)
    print('nussl estimate:\n%s\n\n' % nussl_est)
    print('Errors:\n%s' % error)

    sm = nmf_fit.summary()
    # print( np.dot(W, H))

    x = 1
    y = 2
    z = x + y

if __name__ == '__main__':
    simple_example()
