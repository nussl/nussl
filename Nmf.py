import SeparationBase, AudioSignal
import random, Constants, math
import numpy as np


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
                 distanceMeasure=None, activationUpdateFn=None, templateUpdateFn=None,
                 distanceFn=None):
        super(Nmf, self).__init__()

        if nBases <= 0:
            raise Exception('Need more than 0 bases!')

        if stft.size <= 0:
            raise Exception('STFT size must be > 0!')

        self.stft = stft  # V, in literature
        self.nBases = nBases

        if activationMatrix is None and templateVectors is None:
            self.templateVectors = np.zeros((stft.shape[0], nBases)) # W, in literature
            self.activationMatrix = np.zeros((nBases, stft.shape[1])) # H, in literature
            random.seed(1)
            self.RandomizeInputMatrices()
        else:
            raise NotImplementedError('Cannot do this yet!')

        self.distanceMeasure = DistanceType.DEFAULT
        self.updateType = DistanceType.DEFAULT

        self.shouldUseEpsilon = True # Replace this with something more general
        self.epsilonEuclideanType = True
        self.stoppingEpsilon = 1e-5
        self.maxNumIterations = 20

        self.userUpdateActivation = activationUpdateFn
        self.userUpdateTemplate = templateUpdateFn

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

        shouldStop = False
        nIterations = 0
        while not shouldStop:

            # Update activation matrix
            if self.distanceMeasure == DistanceType.Euclidean:
                self.activationMatrix = self._updateActivationEuclidean()

            elif self.distanceMeasure == DistanceType.Divergence:
                self.activationMatrix = self._updateActivationDivergent()

            else:
                self.activationMatrix = self.userUpdateActivation()

            # Update template vectors
            if self.distanceMeasure == DistanceType.Euclidean:
                self.templateVectors = self._updateTemplateEuclidean()

            elif self.distanceMeasure == DistanceType.Divergence:
                self.templateVectors = self._updateTemplateDivergence()

            else:
                self.templateVectors = self.userUpdateTemplate()

            # Stopping conditions
            nIterations += 1
            if self.shouldUseEpsilon:
                if self.epsilonEuclideanType:
                    print self._euclideanDistance()
                    shouldStop = self._euclideanDistance() <= self.stoppingEpsilon
                else:
                    shouldStop = self._divergence() <= self.stoppingEpsilon
            else:
                shouldStop = nIterations >= self.maxNumIterations

        return self.activationMatrix, self.templateVectors

    def RandomizeInputMatrices(self, shouldNormalize=False):
        self._randomizeMatrix(self.activationMatrix, shouldNormalize)
        self._randomizeMatrix(self.templateVectors, shouldNormalize)

    def _randomizeMatrix(self, M, shouldNormalize=False):
        for i, row in enumerate(M):
            for j, col in enumerate(row):
                M[i][j] = random.random()

                if not shouldNormalize:
                    M[i][j] *= Constants.DEFAULT_MAX_VAL
        return M

    def _updateActivationEuclidean(self):
        # make a new matrix to store results
        activationCopy = np.empty_like(self.activationMatrix)

        # store in memory so we don't have to do n*m calculations.
        templateT = self.templateVectors.T

        # Eq. 4, H update from [1]
        for indices, val in np.ndenumerate(self.activationMatrix):
            result = np.dot(templateT, self.stft)[indices]
            result /= np.dot(np.dot(templateT, self.templateVectors), self.activationMatrix)[indices]
            result *= self.activationMatrix[indices]
            activationCopy[indices] = result

        return activationCopy

    def _updateTemplateEuclidean(self):
        # make a new matrix to store results
        templateCopy = np.empty_like(self.templateVectors)

        # store in memory so we don't have to do n*m calculations.
        activationT = self.activationMatrix.T

        # Eq. 4, W update from [1]
        for indices, val in np.ndenumerate(self.templateVectors):
            result = np.dot(self.stft, activationT)[indices]
            result /= np.dot(np.dot(self.templateVectors, self.activationMatrix), activationT)[indices]
            result *= self.templateVectors[indices]
            templateCopy[indices] = result

        return templateCopy

    def _updateActivationDivergent(self):
        # make a new matrix to store results
        activationCopy = np.empty_like(self.activationMatrix)

        # Eq. 5, H update from [1]
        for indices, val in np.ndenumerate(self.activationMatrix):
            (a, mu) = indices
            result = sum((self.templateVectors[i][a] * self.stft[i][mu])
                         / np.dot(self.templateVectors, self.activationMatrix)[i][mu]
                         for i in range(self.templateVectors.shape[0]))
            result /= sum(self.templateVectors[k][mu] for k in range(self.templateVectors.shape[0]))
            result *= self.activationMatrix[indices]
            activationCopy[indices] = result

        return activationCopy

    def _updateTemplateDivergence(self):
        # make a new matrix to store results
        templateCopy = np.empty_like(self.templateVectors)

        # Eq. 5, W update from [1]
        for indices, val in np.ndenumerate(self.templateVectors):
            (i, a) = indices
            result = sum((self.activationMatrix[a][mu] * self.stft[i][mu])
                         / np.dot(self.templateVectors, self.activationMatrix)[i][mu]
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
        for n in range(self.nBases):
            source = np.dot(self.activationMatrix[n,], self.templateVectors[:,n])
            signal = AudioSignal.AudioSignal(timeSeries=source)
            yield signal


class DistanceType:
    Euclidean = 'Euclidean'
    Divergence = 'Divergence'
    UserDefined = 'UserDefined'
    DEFAULT = Euclidean
