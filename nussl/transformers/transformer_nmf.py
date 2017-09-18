#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Non-negative Matrix Factorization
"""
import math
import numpy as np
import warnings
import matplotlib.pyplot as plt
import librosa


class TransformerNMF(object):
    """
    This is an implementation of the Non-negative Matrix Factorization algorithm for
    general matrix transformations. This implementation receives an input matrix and
    num_components, which defines the number of basis vectors (also called the "dictionary").
    This implementation uses the multiplicative update rules for euclidean distance
    and KL divergence as defined by Lee and Seung in [1].


    References:
    [1] Lee, Daniel D., and H. Sebastian Seung. "Algorithms for non-negative matrix factorization."
        Advances in neural information processing systems. 2001.

    Parameters:
        input_matrix (:obj:`np.array`): The matrix to factor into template and activation matrices (`V`)
        num_components (int): The rank of the resultant factorization matrix
        activation_matrix (:obj:`np.array`): Initial state for the activation matrix
        template_dictionary (:obj:`np.array`): Initial state for the template dictionary (`W`)
        (also called 'components' and 'bases')
        distance_measure (str): Specifies whether to use euclidean or divergence distance metrics (`H`)
        should_update_activation (bool): Whether the activation matrix should be updated for another iteration
        should_update_template (bool): Whether the template matrix should be updated at every iteration
        seed (int): A seed value for the random numbers. If None, no seed is used.
        This will be input to np.random.seed()
        max_num_iterations (int): Maximum number of times that the update rules will be computed
        should_do_epsilon (bool):
        stopping_epsilon (float):

    Attributes:


    Examples:
        :ref:'The Transformer NMF Demo Example <transformer_nmf_demo>'
    """

    # Distance types
    EUCLIDEAN = 'euclidean'
    KL_DIVERGENCE = 'kl_divergence'
    DEFAULT_DISTANCE_TYPE = EUCLIDEAN
    ALL_DISTANCE_TYPES = [EUCLIDEAN, KL_DIVERGENCE]

    def __init__(self, input_matrix, num_components=50,
                 activation_matrix=None, template_dictionary=None, distance_measure=None,
                 should_update_activation=None, should_update_template=None,
                 seed=None, max_num_iterations=50, should_do_epsilon=False, stopping_epsilon=1e10):

        # Check input_matrix
        self._check_input_matrix(input_matrix)

        # It's all good, let's use it!
        self.input_matrix = input_matrix

        # Check num_templates
        if num_components <= 0:
            raise ValueError('Need more than 0 bases!')

        self.num_components = num_components

        # Set the seed value
        if seed is not None:
            np.random.seed(seed)

        # Check activation and templates
        self.activation_matrix = None
        self.template_dictionary = None

        # Initialize templates to random if none provided
        self.template_dictionary = np.random.rand(self.input_matrix.shape[0], num_components) \
            if template_dictionary is None else template_dictionary

        # Initialize activation_matrix to random if none provided
        self.activation_matrix = np.random.rand(num_components, self.input_matrix.shape[1]) \
            if activation_matrix is None else activation_matrix

        # Check to make sure we understand the distance measure
        if distance_measure is not None and distance_measure not in self.ALL_DISTANCE_TYPES:
            raise ValueError('distance_measure is not a known distance type! Known types: {}'
                             .format([t for t in self.ALL_DISTANCE_TYPES]))

        # Initialize distance and update rules
        self.distance_measure = distance_measure if distance_measure is not None else self.DEFAULT_DISTANCE_TYPE

        self.template_update_func = None
        self.activation_update_func = None

        if self._do_euclidean:
            self.template_update_func = self._update_template_euclidean
            self.activation_update_func = self._update_activation_euclidean
        elif self._do_kl_divergence:
            self.template_update_func = self._update_template_kl_divergence
            self.activation_update_func = self._update_activation_kl_divergence

        # Should we update the templates/activations?
        self.should_update_template = True if should_update_template is None else should_update_template
        self.should_update_activation = True if should_update_activation is None else should_update_activation

        # Init other
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

    @property
    def distance(self):
        """
        Calculates the distance between the original matrix (:ref:`input_matrix`) and the dot
        product of the current template (:ref:`templates`) and activation (:ref:`activation_matrix`) matrices using
        the distance type specified by ref:`distance_measure`.
        Returns:

        """
        return self._euclidean_distance() if self._do_euclidean else self._kl_divergence()

    @property
    def _do_euclidean(self):
        return self.distance_measure == self.EUCLIDEAN

    @property
    def _do_kl_divergence(self):
        return self.distance_measure == self.KL_DIVERGENCE

    @property
    def reconstructed_matrix(self):
        """
        PROPERTY
        A reconstruction of the original input_matrix, calculated by doing the dot product of the current values in
        :ref:`templates` and :ref:`activation_matrix`.
        Returns:
            (:obj:`np.ndarray`) of the same shape as :ref:`input_matrix` but containing the dot product of the
            current values in :ref:`templates` and :ref:`activation_matrix`.

        """
        reconstructed_matrix = np.dot(self.template_dictionary, self.activation_matrix)

        if reconstructed_matrix.shape != self.input_matrix.shape:
            raise Exception('Something went wrong! Reconstructed matrix not the same shape as input_matrix!')

        return reconstructed_matrix

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

        if self.num_components is None or self.num_components == 0:
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
            # TODO: Rethink stopping logic
            num_iterations += 1
            if self.should_do_epsilon:
                should_stop = current_distance <= self.stopping_epsilon

            else:
                should_stop = num_iterations >= self.max_num_iterations

        return self.activation_matrix, self.template_dictionary

    def update(self):
        """
        Computes a single update using the update function specified.
        :return: nothing
        """
        # update activation matrix
        if self.should_update_activation:
            self.activation_matrix = self.activation_update_func()

        # update template vectors
        if self.should_update_template:
            self.template_dictionary = self.template_update_func()

    def _update_activation_euclidean(self):
        """
        Computes a new activation matrix using the Lee and Seung multiplicative update algorithm
        :return: An updated activation matrix based on euclidean distance
        """

        # make a new matrix to store results
        activation_copy = np.empty_like(self.activation_matrix)

        # store in memory so we don't have to do n*m calculations.
        template_transpose = self.template_dictionary.T
        temp_transpose_matrix = np.dot(template_transpose, self.input_matrix)
        temp_transpose_act = np.dot(np.dot(template_transpose, self.template_dictionary), self.activation_matrix)

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
        template_copy = np.empty_like(self.template_dictionary)

        # Cache some variables
        activation_transpose = self.activation_matrix.T
        input_matrix_activation_transpose = np.dot(self.input_matrix, activation_transpose)
        temp_act = np.dot(np.dot(self.template_dictionary, self.activation_matrix), activation_transpose)

        # Eq. 4, W update from [1]
        for indices, val in np.ndenumerate(self.template_dictionary):
            result = input_matrix_activation_transpose[indices]
            result /= temp_act[indices]
            result *= self.template_dictionary[indices]
            template_copy[indices] = result

        return template_copy

    def _update_activation_kl_divergence(self):
        """
        Computes a new activation matrix using the Lee and Seung multiplicative update algorithm
        :return: An updated activation matrix based on KL divergence
        """
        # make a new matrix to store results
        activation_copy = np.empty_like(self.activation_matrix)

        dot = np.dot(self.template_dictionary, self.activation_matrix)

        # Eq. 5, H update from [1]
        for indices, val in np.ndenumerate(self.activation_matrix):
            (a, mu) = indices
            result = sum((self.template_dictionary[i][a] * self.input_matrix[i][mu]) / dot[i][mu]
                         for i in range(self.template_dictionary.shape[0]))
            result /= sum(self.template_dictionary[k][a] for k in range(self.template_dictionary.shape[0]))
            result *= self.activation_matrix[indices]
            activation_copy[indices] = result

        return activation_copy

    def _update_template_kl_divergence(self):
        """
        Computes a new template matrix using the Lee and Seung multiplicative update algorithm
        :return: An updated template matrix based on KL divergence
        """
        # make a new matrix to store results
        template_copy = np.empty_like(self.template_dictionary)

        dot = np.dot(self.template_dictionary, self.activation_matrix)

        # Eq. 5, W update from [1]
        for indices, val in np.ndenumerate(self.template_dictionary):
            (i, a) = indices
            result = sum((self.activation_matrix[a][mu] * self.input_matrix[i][mu]) / dot[i][mu]
                         for mu in range(self.activation_matrix.shape[1]))
            result /= sum(self.activation_matrix[a][nu] for nu in range(self.activation_matrix.shape[1]))
            result *= self.template_dictionary[indices]
            template_copy[indices] = result

        return template_copy

    def _euclidean_distance(self):
        """
        Calculates the euclidean distance from the original matrix (:ref:`input_matrix`) to the
        dot product of the current template (:ref:`templates`) and activation (:ref:`activation_matrix`) matrices
        using Euclidean distance
        :return: Euclidean distance
        """
        return sum((self.input_matrix[index] - val) ** 2 for index, val in np.ndenumerate(self.reconstructed_matrix))

    def _kl_divergence(self):
        """
        Calculates the KL divergence between the original matrix (:ref:`input_matrix`) and the
        dot product of the current template (:ref:`templates`) and activation (:ref:`activation_matrix`) matrices.

        :return:

        """
        return sum(
            (self.input_matrix[index] * math.log(self.input_matrix[index] / val, 10) + self.input_matrix[index] - val)
            for index, val in np.ndenumerate(self.reconstructed_matrix))

    MAX_TEMPLATES_FOR_LINES = 30

    def plot(self, output_file, matrix_to_dB=True, title=None, max_y=None, max_x=None, show_divider_lines=None):
        """
        Makes a fancy plot of NMF that shows the original :ref:`input_matrix`, :ref:`activation_matrix`,
        :ref:`template_dictionary`, and :ref:`reconstructed_matrix`.

        Args:
            output_file (string): Path to the output file that will be created.
            matrix_to_dB (bool): Convert the values in all four matrices to dB-spaced values.
            title (string): Title for input matrix
            max_y (int): Max index to show along y-axis (Defaults to whole matrix)
            max_x (int): Max index to show along x-axis (Defaults to whole matrix)
            show_divider_lines (bool): Adds divider lines between activations/templates.
            (Defaults to True if less than :ref:`MAX_TEMPLATES_FOR_LINES` lines.)

        Returns:

        """
        self._check_input_matrix(self.input_matrix)

        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        # below this number, draw lines between each template and activation
        show_divider_lines = show_divider_lines if show_divider_lines is not None \
            else self.num_components <= self.MAX_TEMPLATES_FOR_LINES

        matrix = self.input_matrix
        temp_dict = self.template_dictionary
        activations = self.activation_matrix
        reconstructed_matrix = self.reconstructed_matrix

        x_len, y_len = matrix.shape
        max_y = matrix.shape[1] if max_y is None else max_y
        max_x = matrix.shape[0] if max_x is None else max_x
        min_time = -1
        title = 'Input Matrix' if title is None else title

        # Change to dB if plotting a spectrogram
        if matrix_to_dB:
            matrix = librosa.logamplitude(np.abs(self.input_matrix) ** 2, ref_power=np.max)
            temp_dict = librosa.logamplitude(np.abs(self.template_dictionary) ** 2, ref_power=np.max)
            activations = librosa.logamplitude(np.abs(self.activation_matrix) ** 2, ref_power=np.max)
            reconstructed_matrix = librosa.logamplitude(np.abs(self.reconstructed_matrix) ** 2, ref_power=np.max)

        # Plot the Input Matrix (ax1) and Reconstructed Matrix (ax4)
        matrix_attributes = [{'ax': ax1, 'mat':  matrix, 'title': title},
                             {'ax': ax4, 'mat':  reconstructed_matrix, 'title': 'Reconstructed Matrix'}]

        for p in matrix_attributes:
            ax = p['ax']
            ax.imshow(p['mat'])
            ax.set_ylim(ax.get_ylim()[::-1])
            ax.axis('tight')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title(p['title'])

        # Plot the Template Dictionary on ax2
        ax2.imshow(temp_dict)
        ax2.axis('tight')

        # Major ticks
        ax2.set_xticks(np.arange(0, self.num_components, 1))
        ax2.set_yticks(np.arange(0, y_len, 1))
        ax2.set_ylim([0, max_y])

        # Labels for major ticks
        ax2.set_xticklabels(['' for _ in range(self.num_components)])
        ax2.set_yticklabels(['' for _ in range(y_len)])
        ax2.tick_params(axis=u'both', which=u'both', length=0)

        # Minor ticks
        if show_divider_lines:
            ax2.set_xticks(np.arange(-.5, self.num_components, 1), minor=True)
            ax2.set_yticks(np.arange(-.5, y_len, 1), minor=True)
            ax2.grid(which='minor', axis='x', linestyle='-', color='black', linewidth=1)

        ax2.set_title('Dictionary')

        # Plot the activation matrix on ax3
        ax3.imshow(activations)
        ax3.axis('tight')

        ax3.set_yticks(np.arange(0, self.num_components, 1))
        ax3.set_xticks(np.arange(0, x_len, 1))
        ax3.set_xlim([0, max_x])

        # Labels for major ticks
        ax3.set_xticklabels(['' for _ in range(self.num_components)])
        ax3.set_yticklabels(['' for _ in range(y_len)])
        ax3.tick_params(axis=u'both', which=u'both', length=0)

        # Minor ticks
        if show_divider_lines:
            ax3.set_yticks(np.arange(-.5, self.num_components, 1), minor=True)
            ax3.grid(which='minor', axis='y', linestyle='-', color='black', linewidth=1)

        ax3.set_title('Activations')
        ax3.set_xlim([min_time, activations.shape[1]])

        # Finalize and save
        plt.savefig(output_file)
