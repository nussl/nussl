#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import math
import numpy as np
import warnings

import nussl.constants


class TransformerNMF(object):
    """
    This is an implementation of the Non-negative Matrix Factorization algorithm for
    general matrix transformations. This implementation receives an input matrix and a number,
    num_templates, which defines the number of basis vectors.

    This class provides two implementations of two distance measures, EUCLIDEAN and DIVERGENCE.

    References:
    [1] Lee, Daniel D., and H. Sebastian Seung. "Algorithms for non-negative matrix factorization."
        Advances in neural information processing systems. 2001.

    Parameters:
        input_matrix (np.array): The matrix to factor into template and activation matrices
        num_templates (int): The rank of the templates matrix
        activation_matrix (np.array): An optional seed for the activation matrix
        templates(np.array): An optional seed for the templates matrix
        distance_measure(str): Specifies to use euclidean or divergence distance metrics
        should_update_template(bool): Whether the template matrix should be updated for another iteration
        should_update_activation (bool): Whether the activation matrix should be updated for another iteration
        seed (int): A feed value for the random numbers. If None, no seed is used.
        This will be input to np.random.seed()

    Attributes:


    Examples:
        :ref:'The Transformer NMF Demo Example <transformer_nmf_demo>'
    """

    # Distance types
    EUCLIDEAN = 'euclidean'
    DIVERGENCE = 'divergence'
    DEFAULT_DISTANCE_TYPE = EUCLIDEAN
    ALL_DISTANCE_TYPES = [EUCLIDEAN, DIVERGENCE]

    def __init__(self, input_matrix, num_templates,
                 activation_matrix=None, templates=None, distance_measure=None,
                 should_update_template=None, should_update_activation=None,
                 seed=None, max_num_iterations=50, should_do_epsilon=False, stopping_epsilon=1e10):

        # Check input_matrix
        self._check_input_matrix(input_matrix)

        # It's all good, let's use it!
        self.input_matrix = input_matrix

        # Check num_templates
        if num_templates <= 0:
            raise ValueError('Need more than 0 bases!')

        self.num_templates = num_templates

        # Set the seed value
        if seed is not None:
            np.random.seed(seed)

        # Check activation and templates
        self.activation_matrix = None
        self.templates = None

        # Initialize templates to random if none provided
        self.templates = np.random.rand(self.input_matrix.shape[0], num_templates) if templates is None else templates

        # Initialize activation_matrix to random if none provided
        self.activation_matrix = np.random.rand(num_templates, self.input_matrix.shape[1]) \
            if activation_matrix is None else activation_matrix

        # Check to make sure we understand the
        if distance_measure is not None and distance_measure not in self.ALL_DISTANCE_TYPES:
            raise ValueError('distance_measure is not a known distance type! Known types: {}'
                             .format([t for t in self.ALL_DISTANCE_TYPES]))

        # Initialize other attributes
        self.distance_measure = distance_measure if distance_measure is not None else self.DEFAULT_DISTANCE_TYPE
        self.should_update_template = True if should_update_template is None else should_update_template
        self.should_update_activation = True if should_update_activation is None else should_update_activation

        self.should_do_epsilon = should_do_epsilon
        self.epsilon_euclidean_type = True
        self.stopping_epsilon = stopping_epsilon
        self.max_num_iterations = max_num_iterations

        self.reconstruction_error = []

    @staticmethod
    def _check_input_matrix(matrix):
        if not isinstance(matrix, np.ndarray):
            raise ValueError('input_matrix must be a numpy array!')
        if matrix.size <= 0:
            raise ValueError('Input matrix size must be > 0!')
        if np.iscomplexobj(matrix):
            raise ValueError('Input matrix must be real-valued!')
        if np.min(matrix) < 0.0:
            raise ValueError('Input matrix must be non-negative!')

    def transform(self):
        """
        This runs Non-negative matrix factorization with update rules as outlined in [1].

        Returns:
            * **activation_matrix** (*np.array*) - a 2D numpy matrix containing the estimated activation matrix
            * **templates** (*np.array*) - a 2D numpy matrix containing the estimated templates

        Example:
            ::
            input_matrix = np.random.rand(10, 10)
            nussl_nmf = nussl.TransformerNMF(input_matrix, num_templates=2,
                 activation_matrix=None, templates=None, distance_measure="euclidean",
                 should_update_template=None, should_update_activation=None)

            nussl_nmf.transform()
            signals = nussl_nmf.recombine_calculated_matrices()
        """
        # Check input_matrix
        self._check_input_matrix(self.input_matrix)

        if self.num_templates is None or self.num_templates == 0:
            raise ValueError('Cannot do NMF with no bases!')

        if self.should_do_epsilon:
            warnings.warn('User is expected to have set stopping_epsilon prior to using '
                          'this function. Expect this to take a long time if you have not set '
                          'a suitable epsilon!')

        should_stop = False
        num_iterations = 0
        while not should_stop:

            self.update()

            current_distance = self.distance
            self.reconstruction_error.append(current_distance)

            # Stopping conditions
            num_iterations += 1
            if self.should_do_epsilon:
                should_stop = current_distance <= self.stopping_epsilon

            else:
                should_stop = num_iterations >= self.max_num_iterations

        return self.activation_matrix, self.templates

    def update(self):
        """
        Computes a single update using the update function specified.
        :return: nothing
        """
        # update activation matrix
        if self.should_update_activation:
            if self.distance_measure == self.EUCLIDEAN:
                self.activation_matrix = self._update_activation_euclidean()

            elif self.distance_measure == self.DIVERGENCE:
                self.activation_matrix = self._update_activation_divergence()

        # update template vectors
        if self.should_update_template:
            if self.distance_measure == self.EUCLIDEAN:
                self.templates = self._update_template_euclidean()

            elif self.distance_measure == self.DIVERGENCE:
                self.templates = self._update_template_divergence()

    def _update_activation_euclidean(self):
        """
        Computes a new activation matrix using the Lee and Seung multiplicative update algorithm
        :return: An updated activation matrix based on euclidean distance
        """

        # make a new matrix to store results
        activation_copy = np.empty_like(self.activation_matrix)

        # store in memory so we don't have to do n*m calculations.
        template_transpose = self.templates.T
        temp_transpose_matrix = np.dot(template_transpose, self.input_matrix)
        temp_transpose_act = np.dot(np.dot(template_transpose, self.templates), self.activation_matrix)

        # Eq. 4, H update from [1]
        for indices, val in np.ndenumerate(self.activation_matrix):
            result = temp_transpose_matrix[indices]
            result /= temp_transpose_act[indices]
            result *= self.activation_matrix[indices]
            activation_copy[indices] = result

        return activation_copy

    def _update_template_euclidean(self):
        """
        Computes a new template matrix using the Lee and Seung multiplicative update algorithm
        :return: An updated template matrix based on euclidean distance
        """

        # Make a new matrix to store results
        template_copy = np.empty_like(self.templates)

        # Cache some variables
        activation_transpose = self.activation_matrix.T
        input_matrix_activation_transpose = np.dot(self.input_matrix, activation_transpose)
        temp_act = np.dot(np.dot(self.templates, self.activation_matrix), activation_transpose)

        # Eq. 4, W update from [1]
        for indices, val in np.ndenumerate(self.templates):
            result = input_matrix_activation_transpose[indices]
            result /= temp_act[indices]
            result *= self.templates[indices]
            template_copy[indices] = result

        return template_copy

    def _update_activation_divergence(self):
        """
        Computes a new activation matrix using the Lee and Seung multiplicative update algorithm
        :return: An updated activation matrix based on KL divergence
        """
        # make a new matrix to store results
        activation_copy = np.empty_like(self.activation_matrix)

        dot = np.dot(self.templates, self.activation_matrix)

        # Eq. 5, H update from [1]
        for indices, val in np.ndenumerate(self.activation_matrix):
            (a, mu) = indices
            result = sum((self.templates[i][a] * self.input_matrix[i][mu]) / dot[i][mu]
                         for i in range(self.templates.shape[0]))
            result /= sum(self.templates[k][a] for k in range(self.templates.shape[0]))
            result *= self.activation_matrix[indices]
            activation_copy[indices] = result

        return activation_copy

    def _update_template_divergence(self):
        """
        Computes a new template matrix using the Lee and Seung multiplicative update algorithm
        :return: An updated template matrix based on KL divergence
        """
        # make a new matrix to store results
        template_copy = np.empty_like(self.templates)

        dot = np.dot(self.templates, self.activation_matrix)

        # Eq. 5, W update from [1]
        for indices, val in np.ndenumerate(self.templates):
            (i, a) = indices
            result = sum((self.activation_matrix[a][mu] * self.input_matrix[i][mu]) / dot[i][mu]
                         for mu in range(self.activation_matrix.shape[1]))
            result /= sum(self.activation_matrix[a][nu] for nu in range(self.activation_matrix.shape[1]))
            result *= self.templates[indices]
            template_copy[indices] = result

        return template_copy

    def _euclidean_distance(self):
        """
        Calculates the divergence from the original matrix (:ref:`input_matrix`) to the
        dot product of the current template (:ref:`templates`) and activation (:ref:`activation_matrix`) matrices
        using Euclidean distance
        :return: Euclidean distance
        """
        try:
            mixture = np.dot(self.templates, self.activation_matrix)
        except:
            print(self.activation_matrix.shape, self.templates.shape)
            return

        if mixture.shape != self.input_matrix.shape:
            raise Exception('Something went wrong! Recombining the activation matrix '
                            'and template vectors is not the same size as the input matrix!')

        return sum((self.input_matrix[index] - val) ** 2 for index, val in np.ndenumerate(mixture))

    def _divergence(self):
        """
        Calculates the divergence from the original matrix (:ref:`input_matrix`) to the
        dot product of the current template (:ref:`templates`) and activation (:ref:`activation_matrix`) matrices
        using KL Divergence

        :return:

        """
        mixture = np.dot(self.activation_matrix, self.templates)

        if mixture.shape != self.input_matrix.shape:
            raise Exception('Something went wrong! Recombining the activation matrix '
                            'and template vectors is not the same size as the input matrix!')

        return sum(
            (self.input_matrix[index] * math.log(self.input_matrix[index] / val, 10) + self.input_matrix[index] - val)
            for index, val in np.ndenumerate(mixture))

    @property
    def distance(self):
        """
        Calculates the distance between the original matrix (:ref:`input_matrix`) and the dot
        product of the current template (:ref:`templates`) and activation (:ref:`activation_matrix`) matrices using
        the distance type specified by ref:`distance_measure`.
        Returns:

        """
        return self._euclidean_distance() if self.epsilon_euclidean_type else self._divergence()

    # def recombine_calculated_matrices(self):
    #     new_matrices = []
    #     for n in range(self.num_templates):
    #         matrix = np.empty_like(self.activation_matrix)
    #         matrix[n, ] = self.activation_matrix[n,]
    #
    #         new_matrix = np.dot(self.templates, matrix)
    #         new_matrices.append(new_matrix)
    #     return new_matrices

    def plot(self, output_file, **kwargs):
        raise NotImplementedError('Sorry, you cannot do this yet.')
