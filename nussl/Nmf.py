#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import math
import numpy as np

import separation_base
import audio_signal
import constants


class NMF(separation_base.SeparationBase):
    """
    This is an implementation of the Non-negative Matrix Factorization algorithm for
    source separation. This implementation receives an audio_signal object
    and a number, num_templates, which defines the number of bases vectors.

    This class provides two implementations of distance measures, EUCLIDEAN and DIVERGENCE,
    and also allows the user to define distance measure function.

    References:
    [1] Lee, Daniel D., and H. Sebastian Seung. "Algorithms for non-negative matrix factorization."
        Advances in neural information processing systems. 2001.
    """

    def __str__(self):
        return "Nmf"

    # TODO: Change this so that NMF accepts an AudioSignal object and not a raw stft
    def __init__(self, input_audio_signal, num_templates,
                 activation_matrix=None, templates=None, distance_measure=None,
                 should_update_template=None, should_update_activation=None):
        self.__dict__.update(locals())
        super(NMF, self).__init__(input_audio_signal=input_audio_signal)

        if num_templates <= 0:
            raise Exception('Need more than 0 bases!')

        self.stft = self.audio_signal.stft(overwrite=False) # V in literature
        self.num_templates = num_templates

        if self.stft.size <= 0:
            raise Exception('STFT size must be > 0!')

        if activation_matrix is None and templates is None:
            self.templates = np.zeros((self.stft.shape[0], num_templates))  # W, in literature
            self.activation_matrix = np.zeros((num_templates, self.stft.shape[1]))  # H, in literature
            self.randomize_input_matrices()
        elif activation_matrix is not None and templates is not None:
            self.templates = templates
            self.activation_matrix = activation_matrix
        else:
            raise Exception('Must provide both activation matrix and template vectors or nothing at all!')

        self.distance_measure = distance_measure if distance_measure is not None else DistanceType.DEFAULT
        self.should_update_template = True if should_update_template is None else should_update_template
        self.should_update_activation = True if should_update_activation is None else should_update_activation

        self.should_use_epsilon = False  # Replace this with something more general
        self.epsilon_euclidean_type = True
        self.stopping_epsilon = 1e10
        self.max_num_iterations = 20

    def run(self):
        """
        This runs the NMF separation algorithm. This function assumes that all
        parameters have been set prior to running.

        No inputs. do_STFT and N must be set prior to calling this function.

        Returns an activation matrix (in a 2d numpy array)
        and a set of template vectors (also 2d numpy array).
        """

        if self.stft is None or self.stft.size == 0:
            raise Exception('Cannot do NMF with an empty STFT!')

        if self.num_templates is None or self.num_templates == 0:
            raise Exception('Cannot do NMF with no bases!')

        if self.should_use_epsilon:
            print('Warning: User is expected to have set stopping_epsilon prior to using'
                  ' this function. Expect this to take a long time if you have not set'
                  ' a suitable epsilon.')

        should_stop = False
        num_iterations = 0
        while not should_stop:

            self.update()

            # Stopping conditions
            num_iterations += 1
            if self.should_use_epsilon:
                if self.epsilon_euclidean_type:
                    should_stop = self._euclidean_distance() <= self.stopping_epsilon
                else:
                    should_stop = self._divergence() <= self.stopping_epsilon
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
            if self.distance_measure == DistanceType.EUCLIDEAN:
                self.activation_matrix = self._update_activation_euclidean()

            elif self.distance_measure == DistanceType.DIVERGENCE:
                self.activation_matrix = self._update_activation_divergent()

        # update template vectors
        if self.should_update_template:
            if self.distance_measure == DistanceType.EUCLIDEAN:
                self.templates = self._update_template_euclidean()

            elif self.distance_measure == DistanceType.DIVERGENCE:
                self.templates = self._update_template_divergence()

    def _update_activation_euclidean(self):
        # make a new matrix to store results
        activation_copy = np.empty_like(self.activation_matrix)

        # store in memory so we don't have to do n*m calculations.
        template_T = self.templates.T
        temp_T_stft = np.dot(template_T, self.stft)
        temp_T_act = np.dot(np.dot(template_T, self.templates), self.activation_matrix)

        # Eq. 4, H update from [1]
        for indices, val in np.ndenumerate(self.activation_matrix):
            result = temp_T_stft[indices]
            result /= temp_T_act[indices]
            result *= self.activation_matrix[indices]
            activation_copy[indices] = result

        return activation_copy

    def _update_template_euclidean(self):
        # make a new matrix to store results
        template_copy = np.empty_like(self.templates)

        # store in memory so we don't have to do n*m calculations.
        activation_T = self.activation_matrix.T
        stft_act_T = np.dot(self.stft, activation_T)
        temp_act = np.dot(np.dot(self.templates, self.activation_matrix), activation_T)

        # Eq. 4, W update from [1]
        for indices, val in np.ndenumerate(self.templates):
            result = stft_act_T[indices]
            result /= temp_act[indices]
            result *= self.templates[indices]
            template_copy[indices] = result

        return template_copy

    def _update_activation_divergent(self):
        # make a new matrix to store results
        activation_copy = np.empty_like(self.activation_matrix)

        dot = np.dot(self.templates, self.activation_matrix)

        # Eq. 5, H update from [1]
        for indices, val in np.ndenumerate(self.activation_matrix):
            (a, mu) = indices
            result = sum((self.templates[i][a] * self.stft[i][mu]) / dot[i][mu]
                         for i in range(self.templates.shape[0]))
            result /= sum(self.templates[k][a] for k in range(self.templates.shape[0]))
            result *= self.activation_matrix[indices]
            activation_copy[indices] = result

        return activation_copy

    def _update_template_divergence(self):
        # make a new matrix to store results
        template_copy = np.empty_like(self.templates)

        dot = np.dot(self.templates, self.activation_matrix)

        # Eq. 5, W update from [1]
        for indices, val in np.ndenumerate(self.templates):
            (i, a) = indices
            result = sum((self.activation_matrix[a][mu] * self.stft[i][mu]) / dot[i][mu]
                         for mu in range(self.activation_matrix.shape[1]))
            result /= sum(self.activation_matrix[a][nu] for nu in range(self.activation_matrix.shape[1]))
            result *= self.templates[indices]
            template_copy[indices] = result

        return template_copy

    def _euclidean_distance(self):
        try:
            mixture = np.dot(self.templates, self.activation_matrix)
        except:
            print(self.activation_matrix.shape, self.templates.shape)
            return

        if mixture.shape != self.stft.shape:
            raise Exception('Something went wrong! Recombining the activation matrix '
                            'and template vectors is not the same size as the STFT!')

        return sum((self.stft[index] - val) ** 2 for index, val in np.ndenumerate(mixture))

    def _divergence(self):
        mixture = np.dot(self.activation_matrix, self.templates)

        if mixture.shape != self.stft.shape:
            raise Exception('Something went wrong! Recombining the activation matrix '
                            'and template vectors is not the same size as the STFT!')

        return sum(
            (self.stft[index] * math.log(self.stft[index] / val, 10) + self.stft[index] - val)
            for index, val in np.ndenumerate(mixture))

    def make_audio_signals(self):
        # TODO: this!
        raise NotImplementedError('This does not work yet.')
        signals = []
        for stft in self.recombine_calculated_matrices():
            signal = audio_signal.AudioSignal(stft=stft)
            signal.istft()
            signals.append(signal)
        return signals

    def recombine_calculated_matrices(self):
        new_matrices = []
        for n in range(self.num_templates):
            matrix = np.empty_like(self.activation_matrix)
            matrix[n, ] = self.activation_matrix[n, ]

            new_stft = np.dot(self.templates, matrix)
            new_matrices.append(new_stft)
        return new_matrices

    def randomize_input_matrices(self, shouldNormalize=False):
        self._randomize_matrix(self.activation_matrix, shouldNormalize)
        self._randomize_matrix(self.templates, shouldNormalize)

    @staticmethod
    def _randomize_matrix(M, shouldNormalize=False):
        for i, row in enumerate(M):
            for j, col in enumerate(row):
                M[i][j] = random.random()

                if not shouldNormalize:
                    M[i][j] *= constants.DEFAULT_MAX_VAL
        return M

    def plot(self, outputFile, **kwargs):
        raise NotImplementedError('Sorry, you cannot do this yet.')


class DistanceType:
    EUCLIDEAN = 'euclidean'
    DIVERGENCE = 'divergence'
    DEFAULT = EUCLIDEAN

    def __init__(self):
        pass
