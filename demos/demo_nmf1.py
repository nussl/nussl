import time
import numpy as np
import nussl

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
    nBases = 2
    nmf = nussl.NMF(nBases)
    nmf.should_use_epsilon = False
    nmf.max_num_iterations = 100
    nmf.distance_measure = nussl.DistanceType.EUCLIDEAN

    # run NU NMF
    start = time.time()
    nmf.run()
    print '{0:.3f}'.format(time.time() - start), 'seconds for NUSSL'

    print 'Original mixture =\n', mixture
    print 'NMF Reconstructed mixture =\n', np.dot(nmf.templates, nmf.activation_matrix)

    print 'Sources ='
    signals = nmf.recombine_calculated_matrices()
    for sig in signals:
        print sig


if __name__ == '__main__':
    simple_example()
