#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import nussl

import nimfa


class TransformerNMFUnitTests(unittest.TestCase):

    n_attempts = 2
    max_error_pct = 0.05
    n_bases = 2
    n_iters = 100

    def test_small_mixture_matrix(self):
        """
        A simple example of NMF using euclidean and divergence in nussl with a 4 by 4 matrix
        """
        # Make two simple matrices
        for n in range(4, 20, 4):
            a = np.arange(n ** 2).reshape((n, n))
            b = np.add(np.multiply(2., a), 3.)

            # Mix them together
            mixture = np.dot(b, a)

            # Run with euclidean distance
            dist_type = nussl.transformers.TransformerNMF.EUCLIDEAN
            self.calculate_nmf_error(mixture, self.n_bases, dist_type, self.n_iters, self.n_attempts, n)

            # Run with divergence
            dist_type = nussl.transformers.TransformerNMF.KL_DIVERGENCE
            self.calculate_nmf_error(mixture, self.n_bases, dist_type, self.n_iters, self.n_attempts, n)


    def test_random_matrix(self):
        for n in range(4, 10, 2):
            matrix = np.random.rand(n, n)

            # Run on euclidean
            distance_type = nussl.transformers.TransformerNMF.EUCLIDEAN
            self.calculate_nmf_error(matrix, self.n_bases, distance_type, self.n_iters, self.n_attempts, n)

            # Run on divergence
            distance_type = nussl.transformers.TransformerNMF.KL_DIVERGENCE
            self.calculate_nmf_error(matrix, self.n_bases, distance_type, self.n_iters, self.n_attempts, n)

    def calculate_nmf_error(self, mixture, n_bases, dist_type, iterations, attempts, seed):
        div = nussl.transformers.TransformerNMF.KL_DIVERGENCE
        nimfa_type = 'divergence' if dist_type == div else dist_type

        for i in range(attempts):
            # Set up nussl NMF
            nussl_nmf = nussl.TransformerNMF(mixture, n_bases, max_num_iterations=iterations,
                                             distance_measure=dist_type, seed=seed)
            # Run nussl NMF
            nussl_nmf.transform()

            # Set up nimfa NMF
            nimfa_nmf = nimfa.Nmf(mixture, max_iter=iterations, rank=n_bases, update=nimfa_type,
                                  W=nussl_nmf.template_dictionary,
                                  H=nussl_nmf.activation_matrix)  # init to same matrices as nussl

            # Run nimfa NMF
            nmf_fit = nimfa_nmf()

            # Dot the results
            nimfa_est = np.dot(nmf_fit.basis(), nmf_fit.coef())
            nussl_est = np.dot(nussl_nmf.template_dictionary, nussl_nmf.activation_matrix)

            # calculate errors
            max_nussl_error = np.max(np.abs(nussl_est - mixture) / mixture)
            max_nimfa_error = np.max(np.abs(nimfa_est - mixture) / mixture)
            max_diff = max_nussl_error - max_nimfa_error

            # IF nussl's max error is bigger than nimfa's
            # AND nussl's max error bigger than the specified max error (0.05, or 5%)
            # AND the difference between the max errors is larger than 0.05
            # THEN we throw an exception
            # i.e., make sure nussl's results are close to nimfa's
            if max_nussl_error > max_nimfa_error \
                    and max_nussl_error > self.max_error_pct \
                    and max_diff > self.max_error_pct:
                raise Exception('max nussl error is larger than nimfa and self.max_error_pct')
