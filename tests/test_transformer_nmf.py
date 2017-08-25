#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import nussl

import nimfa


class TransformerNMFUnitTests(unittest.TestCase):

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

            # Set up NMFs and run on euclidean while calculating error
            n_bases = 2
            iterations = 100
            # Set error parameters
            num_std_deviations = 5
            percent_over_std_dev = .05

            # Run with euclidean distance
            distance_type = 'euclidean'
            errors = self.calculate_nmf_error(mixture, n_bases, distance_type, iterations)
            error_check_euclidean = self.analyze_errors(errors, iterations, num_std_deviations, percent_over_std_dev)
            assert error_check_euclidean

            # Run with divergence
            distance_type = 'divergence'
            errors = self.calculate_nmf_error(mixture, n_bases, distance_type, iterations)
            error_check_divergence = self.analyze_errors(errors, iterations, num_std_deviations, percent_over_std_dev)
            assert error_check_divergence

    def test_random_matrix(self):
        for n in range(4, 10, 2):
            matrix = np.random.rand(n, n)
            # Set up NMFs and run on euclidean while calculating error
            n_bases = 2
            iterations = 100
            # Set error parameters
            num_std_deviations = 10
            percent_over_std_dev = .50

            # Run on euclidean
            distance_type = "euclidean"
            errors = self.calculate_nmf_error(matrix, n_bases, distance_type, iterations)
            error_check_euclidean = self.analyze_errors(errors, iterations, num_std_deviations, percent_over_std_dev)

            # Run on divergence
            distance_type = "divergence"
            errors = self.calculate_nmf_error(matrix, n_bases, distance_type, iterations)
            error_check_divergence = self.analyze_errors(errors, iterations, num_std_deviations, percent_over_std_dev)

            assert error_check_euclidean
            assert error_check_divergence

    @staticmethod
    def calculate_nmf_error(mixture, n_bases, type, iterations):
        nussl_nmf = nussl.TransformerNMF(mixture, n_bases)
        nussl_nmf.should_do_epsilon = False
        nussl_nmf.max_num_iterations = iterations
        if type == "euclidean":
            nussl_nmf.distance_measure = nussl.transformers.TransformerNMF.EUCLIDEAN
        elif type == "divergence":
            nussl_nmf.distance_measure = nussl.transformers.TransformerNMF.DIVERGENCE
        else:
            return "Use either euclidean or divergence!"

        # set up nimfa NMF
        matrix = mixture
        nimfa_nmf = nimfa.Nmf(matrix, max_iter=iterations, rank=n_bases, update=type)

        # calculate errors
        errors = []
        for i in range(iterations):
            # run NU NMF
            nussl_nmf.transform()
            # signals = nussl_nmf.recombine_calculated_matrices()

            # run nimfa NMF
            nmf_fit = nimfa_nmf()
            W = nmf_fit.basis()
            H = nmf_fit.coef()

            nimfa_est = np.dot(W, H)
            nussl_est = np.dot(nussl_nmf.templates, nussl_nmf.activation_matrix)
            error = np.abs(nussl_est - nimfa_est) / nimfa_est * 100
            errors.append(error.flatten())
        return errors

    @staticmethod
    def analyze_errors(errors, iterations, num_std_deviations, percent_over_std_dev):
        # analyze mean and std deviation of errors
        errors = np.array(errors).flatten()
        mean_error = np.mean(errors)
        std_dev_error = np.std(errors)
        std_check = True
        num_over_std_dev = 0

        for error in errors:
            if error > (mean_error + (num_std_deviations * std_dev_error)):
                num_over_std_dev += 1

        if (num_over_std_dev / float(iterations)) > percent_over_std_dev:
            std_check = False

        return std_check
