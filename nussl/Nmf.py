import random
import math

import numpy as np

import SeparationBase
import AudioSignal
import Constants


class Nmf(SeparationBase.SeparationBase):
    """
    This is an implementation of the Non-negative Matrix Factorization algorithm for
    source separation. This implementation cannot receive a raw audio signal, rather
    it only accepts a STFT and a number, nBases, which defines the number of bases
    vectors.

    This class provides two implementations of distance measures, Euclidean and Divergence,
    and also allows the user to define distance measure function.

    References:
    [1] Lee, Daniel D., and H. Sebastian Seung. "Algorithms for non-negative matrix factorization."
        Advances in neural information processing systems. 2001.
    """

    def __str__(self):
        return "Nmf"

    def __init__(self, stft, nBases, activationMatrix=None, templateVectors=None,
                 distanceMeasure=None, shouldUpdateTemplate=None, shouldUpdateActivation=None):
        self.__dict__.update(locals())
        super(Nmf, self).__init__()

        if nBases <= 0:
            raise Exception('Need more than 0 bases!')

        if stft.size <= 0:
            raise Exception('STFT size must be > 0!')

        self.stft = stft  # V, in literature
        self.nBases = nBases

        if activationMatrix is None and templateVectors is None:
            self.templateVectors = np.zeros((stft.shape[0], nBases))  # W, in literature
            self.activationMatrix = np.zeros((nBases, stft.shape[1]))  # H, in literature
            self.RandomizeInputMatrices()
        elif activationMatrix is not None and templateVectors is not None:
            self.templateVectors = templateVectors
            self.activationMatrix = activationMatrix
        else:
            raise Exception('Must provide both activation matrix and template vectors or nothing at all!')

        self.distanceMeasure = distanceMeasure if distanceMeasure is not None else DistanceType.DEFAULT
        self.ShouldUpdateTemplate = True if shouldUpdateTemplate is None else shouldUpdateTemplate
        self.ShouldUpdateActivation = True if shouldUpdateActivation is None else shouldUpdateActivation

        self.shouldUseEpsilon = False  # Replace this with something more general
        self.epsilonEuclideanType = True
        self.stoppingEpsilon = 1e10
        self.maxNumIterations = 20



    def Run(self):
        """
        This runs the NMF separation algorithm. This function assumes that all
        parameters have been set prior to running.

        No inputs. STFT and N must be set prior to calling this function.

        Returns an activation matrix (in a 2d numpy array)
        and a set of template vectors (also 2d numpy array).
        """

        if self.stft is None or self.stft.size == 0:
            raise Exception('Cannot do NMF with an empty STFT!')

        if self.nBases is None or self.nBases == 0:
            raise Exception('Cannot do NMF with no bases!')

        if self.shouldUseEpsilon:
            print 'Warning: User is expected to have set stoppingEpsilon prior to using' \
                  ' this function. Expect this to take a long time if you have not set' \
                  ' a suitable epsilon.'

        shouldStop = False
        nIterations = 0
        while not shouldStop:

            self.Update()

            # Stopping conditions
            nIterations += 1
            if self.shouldUseEpsilon:
                if self.epsilonEuclideanType:
                    shouldStop = self._euclideanDistance() <= self.stoppingEpsilon
                else:
                    shouldStop = self._divergence() <= self.stoppingEpsilon
            else:
                shouldStop = nIterations >= self.maxNumIterations

        return self.activationMatrix, self.templateVectors

    def Update(self):
        """
        Computes a single update using the update function specified.
        :return: nothing
        """
        # Update activation matrix
        if self.ShouldUpdateActivation:
            if self.distanceMeasure == DistanceType.Euclidean:
                self.activationMatrix = self._updateActivationEuclidean()

            elif self.distanceMeasure == DistanceType.Divergence:
                self.activationMatrix = self._updateActivationDivergent()

        # Update template vectors
        if self.ShouldUpdateTemplate:
            if self.distanceMeasure == DistanceType.Euclidean:
                self.templateVectors = self._updateTemplateEuclidean()

            elif self.distanceMeasure == DistanceType.Divergence:
                self.templateVectors = self._updateTemplateDivergence()

    def _updateActivationEuclidean(self):
        # make a new matrix to store results
        activationCopy = np.empty_like(self.activationMatrix)

        # store in memory so we don't have to do n*m calculations.
        templateT = self.templateVectors.T
        temp_T_stft = np.dot(templateT, self.stft)
        temp_T_act = np.dot(np.dot(templateT, self.templateVectors), self.activationMatrix)

        # Eq. 4, H update from [1]
        for indices, val in np.ndenumerate(self.activationMatrix):
            result = temp_T_stft[indices]
            result /= temp_T_act[indices]
            result *= self.activationMatrix[indices]
            activationCopy[indices] = result

        return activationCopy

    def _updateTemplateEuclidean(self):
        # make a new matrix to store results
        templateCopy = np.empty_like(self.templateVectors)

        # store in memory so we don't have to do n*m calculations.
        activationT = self.activationMatrix.T
        stft_act_T = np.dot(self.stft, activationT)
        temp_act = np.dot(np.dot(self.templateVectors, self.activationMatrix), activationT)

        # Eq. 4, W update from [1]
        for indices, val in np.ndenumerate(self.templateVectors):
            result = stft_act_T[indices]
            result /= temp_act[indices]
            result *= self.templateVectors[indices]
            templateCopy[indices] = result

        return templateCopy

    def _updateActivationDivergent(self):
        # make a new matrix to store results
        activationCopy = np.empty_like(self.activationMatrix)

        dot = np.dot(self.templateVectors, self.activationMatrix)

        # Eq. 5, H update from [1]
        for indices, val in np.ndenumerate(self.activationMatrix):
            (a, mu) = indices
            result = sum((self.templateVectors[i][a] * self.stft[i][mu]) / dot[i][mu]
                         for i in range(self.templateVectors.shape[0]))
            result /= sum(self.templateVectors[k][a] for k in range(self.templateVectors.shape[0]))
            result *= self.activationMatrix[indices]
            activationCopy[indices] = result

        return activationCopy

    def _updateTemplateDivergence(self):
        # make a new matrix to store results
        templateCopy = np.empty_like(self.templateVectors)

        dot = np.dot(self.templateVectors, self.activationMatrix)

        # Eq. 5, W update from [1]
        for indices, val in np.ndenumerate(self.templateVectors):
            (i, a) = indices
            result = sum((self.activationMatrix[a][mu] * self.stft[i][mu]) / dot[i][mu]
                         for mu in range(self.activationMatrix.shape[1]))
            result /= sum(self.activationMatrix[a][nu] for nu in range(self.activationMatrix.shape[1]))
            result *= self.templateVectors[indices]
            templateCopy[indices] = result

        return templateCopy

    def _euclideanDistance(self):
        try:
            mixture = np.dot(self.templateVectors, self.activationMatrix)
        except:
            print self.activationMatrix.shape, self.templateVectors.shape
            return

        if mixture.shape != self.stft.shape:
            raise Exception('Something went wrong! Recombining the activation matrix '
                            'and template vectors is not the same size as the STFT!')

        return sum((self.stft[index] - val) ** 2 for index, val in np.ndenumerate(mixture))

    def _divergence(self):
        mixture = np.dot(self.activationMatrix, self.templateVectors)

        if mixture.shape != self.stft.shape:
            raise Exception('Something went wrong! Recombining the activation matrix '
                            'and template vectors is not the same size as the STFT!')

        return sum(
            (self.stft[index] * math.log(self.stft[index] / val, 10) + self.stft[index] - val)
            for index, val in np.ndenumerate(mixture))

    def MakeAudioSignals(self):
        raise NotImplementedError('This does not work yet.')
        signals = []
        for stft in self.RecombineCalculatedMatrices():
            signal = AudioSignal.AudioSignal(stft=stft)
            signal.iSTFT()
            signals.append(signal)
        return signals

    def RecombineCalculatedMatrices(self):
        newMatrices = []
        for n in range(self.nBases):
            matrix = np.empty_like(self.activationMatrix)
            matrix[n,] = self.activationMatrix[n,]

            newStft = np.dot(self.templateVectors, matrix)
            newMatrices.append(newStft)
        return newMatrices

    def RandomizeInputMatrices(self, shouldNormalize=False):
        self._randomizeMatrix(self.activationMatrix, shouldNormalize)
        self._randomizeMatrix(self.templateVectors, shouldNormalize)

    @staticmethod
    def _randomizeMatrix(M, shouldNormalize=False):
        for i, row in enumerate(M):
            for j, col in enumerate(row):
                M[i][j] = random.random()

                if not shouldNormalize:
                    M[i][j] *= Constants.DEFAULT_MAX_VAL
        return M

    def Plot(self, outputFile):
        raise NotImplementedError('Sorry, you cannot do this yet.')


class DistanceType:
    Euclidean = 'Euclidean'
    Divergence = 'Divergence'
    UserDefined = 'UserDefined'
    DEFAULT = Euclidean
    def __init__(self):
        pass
