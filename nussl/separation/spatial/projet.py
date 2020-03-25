import copy

import numpy as np
import torch

from .. import SeparationBase, SeparationException
from ... import AudioSignal


class Projet(SeparationBase):
    """
    Implements the PROJET algorithm for spatial audio separation using projections.
    This implementation uses PyTorch to speed up computation considerably. PROJET
    does the following steps:

    1. Project the complex stereo STFT onto multiple angles and delay via
       projection and delay matrix transformations.
    2. Initialize the parameters of the system to "remix" these projections along with
       PSDs of the sources such that they try to reconstruct the original stereo mixture.
    3. Find the optimal parameters via multiplicative update rules for P and for Q.
    4. Use the discovered parameters to isolate the sources via spatial cues.

    This implementation considers BOTH panning and delays when isolating sources.
    PROJET is not a masking based method, it estimates the sources directly by
    projecting the complex STFT.

    Args:
          input_audio_signal (AudioSignal): Audio signal to separate.
          num_sources (int): Number of source to separate.
          estimates (list of AudioSignal):  initial estimates for the separated sources
            if available. These will be used to initialize the update algorithm. So
            one could (for example), run FT2D on a signal and then refine the estimates
            using PROJET. Defaults to None (randomly initialize P).
          num_iterations (int, optional): Number of iterations to do for the update 
            rules for P and Q. Defaults to 50.
          maximum_delay_in_samples (int, optional): Maximum delay in samples that you are
            willing to consider in the projection matrices. Defaults to 20.
          location_set_panning (int, optional): How many locations in panning you are 
            willing to consider. Defaults to 30.
          location_set_delay (int, optional): How many delays you are willing to 
            consider. Defaults to 17.
          projection_set_panning (int, optional): How many projections you are willing
            use in panning-space. Defaults to 10.
          projection_set_delay (int, optional): How many delays you are willing to project
            the mixutre onto in panning-space. Defaults to 9.
          beta (int, optional):  Beta in beta divergence. See Table 1 in [1]. Defaults to 1.
          alpha (int, optional): Power to raise each power spectral density estimate of each
            source to. Defaults to 1.
          device (str, optional): Device to use when performing update rules. 'cuda' will
            be fastest, if available. Defaults to 'cpu'.

    References:

    [1] Fitzgerald, Derry, Antoine Liutkus, and Roland Badeau. 
        "Projection-based demixing of spatial audio." 
        IEEE/ACM Transactions on Audio, Speech, and Language 
        Processing 24.9 (2016): 1560-1572.

    [2] Fitzgerald, Derry, Antoine Liutkus, and Roland Badeau. 
        "Projet—spatial audio separation using projections." 2016 IEEE International 
        Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2016.
    """

    def __init__(self, input_audio_signal, num_sources, estimates=None, num_iterations=50,
                 maximum_delay_in_samples=20, location_set_panning=30, location_set_delay=17,
                 projection_set_panning=10, projection_set_delay=9, beta=1, alpha=1, device='cpu'):

        self.num_sources = num_sources
        self.alpha = alpha
        self.beta = beta
        self.num_iterations = num_iterations
        self.eps = 1e-8
        self.device = device
        self.projection_set_panning = projection_set_panning
        self.projection_set_delay = projection_set_delay
        self.location_set_panning = location_set_panning
        self.location_set_delay = location_set_delay
        self.maximum_delay_in_samples = maximum_delay_in_samples
        self.projection_set = None
        self.inverse_projection_set = None
        self.reconstructions = None

        super().__init__(input_audio_signal=input_audio_signal)

        if estimates is None:
            self.P = None
        else:
            if len(estimates) != self.num_sources:
                raise SeparationException(
                    "Number of estimates must be equal to num_sources!")
            psds = []
            for e in estimates:
                _e = copy.deepcopy(e)
                _e.to_mono()
                _e.stft_params = self.audio_signal.stft_params
                psds.append(np.abs(_e.stft()))

            self.P = np.stack(psds, axis=-1)

    def _preprocess_audio_signal(self):
        self.stft = self.audio_signal.stft()
        F, T, I = self.stft.shape

        self.device = 'cpu' if not torch.cuda.is_available() else self.device

        pannings = np.linspace(0, np.pi / 2, self.location_set_panning)
        delays = np.linspace(
            -self.maximum_delay_in_samples,
            self.maximum_delay_in_samples,
            self.location_set_delay
        )
        self.location_set = self.create_panning_delay_set(pannings, delays, F, I)

        pannings = np.linspace(-np.pi / 2, 0, self.projection_set_panning)
        delays = np.linspace(
            -self.maximum_delay_in_samples,
            self.maximum_delay_in_samples,
            self.projection_set_delay
        )
        self.projection_set = self.create_panning_delay_set(pannings, delays, F, I)
        self.inverse_projection_set = np.linalg.pinv(self.projection_set)

        self.V, self.complex_projections = self.create_projections()
        self.K = self.create_k_matrix()

    @staticmethod
    def create_panning_delay_set(pannings, delays, F, I):
        panning_delay_set = np.zeros(
            (F, len(pannings), len(delays), I), dtype='complex'
        )

        for i, delay in enumerate(delays):
            phase_change = np.exp(
                -1j * 2 * np.pi * np.linspace(0, 0.5, F) * delay
            )
            panning_delay_set[:, :, i, 0] = np.outer(
                np.ones(F), np.cos(pannings)
            )
            panning_delay_set[:, :, i, 1] = np.outer(
                phase_change, np.sin(pannings)
            )
        return panning_delay_set

    def create_projections(self):
        F = self.stft.shape[0]
        inner_dim = self.projection_set.shape[-1]
        V = (
                self.projection_set.reshape(F, -1, inner_dim) @
                self.stft.reshape(F, -1, inner_dim).transpose(0, 2, 1)
        )
        V = V.reshape((F,) + self.projection_set.shape[1:-1] + (-1,))
        return self._convert_to_tensor(np.abs(V) ** self.alpha), V

    def _convert_to_tensor(self, data):
        tensor = torch.from_numpy(data)
        return tensor.to(self.device)

    @staticmethod
    def _convert_to_numpy(data):
        array = data.cpu().data.numpy()
        return array

    def create_k_matrix(self):
        F = self.stft.shape[0]
        inner_dim = self.location_set.shape[-1]

        K = (
                self.projection_set.reshape(F, -1, inner_dim) @
                self.location_set.reshape(F, -1, inner_dim).transpose(0, 2, 1)
        )

        K = np.abs(K.reshape(
            (F,) + self.projection_set.shape[1:-1] + self.location_set.shape[1:-1])) ** self.alpha
        return self._convert_to_tensor(K)

    def initialize_parameters(self):
        F, T, I = self.stft.shape
        P = np.abs(np.random.randn(F, T, self.num_sources))
        Q = np.abs(np.random.randn(*self.location_set.shape[1:3], self.num_sources))
        return self._convert_to_tensor(P), self._convert_to_tensor(Q)

    def _update_sigma(self, P, Q, KQ):
        F = self.stft.shape[0]
        inner_dim = KQ.shape[-1]
        sigma = (
                KQ.reshape(F, -1, inner_dim) @
                P.transpose(2, 1)
        )

        sigma = sigma.reshape(
            P.shape[0], KQ.shape[1], KQ.shape[2], P.shape[1])
        return sigma

    def _update_P(self, P, sigma, KQ):
        F = self.stft.shape[0]
        temps = [
            (sigma ** (self.beta - 2)) * self.V,
            sigma ** (self.beta - 1)
        ]
        inner_dim = KQ.shape[1] * KQ.shape[2]
        P_num_denom = [
            self.eps + (
                    KQ.reshape(F, inner_dim, -1).transpose(2, 1) @
                    temp.reshape(F, inner_dim, -1)
            )
            for temp in temps
        ]
        P_update = (P_num_denom[0] / P_num_denom[1]).transpose(2, 1)
        return P * P_update

    def _update_Q(self, P, sigma, Q):
        F = self.stft.shape[0]
        temps = [
            (sigma ** (self.beta - 2)) * self.V,
            sigma ** (self.beta - 1)
        ]
        inner_dim = self.K.shape[1] * self.K.shape[2]
        Q_num_denom = [
            self.K.reshape(F, inner_dim, -1).transpose(2, 1) @
            (temp.reshape(F, -1, P.shape[1]) @ P)
            for temp in temps
        ]
        Q_num_denom = [
            x.reshape(F, *Q.shape).sum(dim=0) for x in Q_num_denom
        ]
        Q_update = Q_num_denom[0] / Q_num_denom[1]
        return Q * Q_update

    def _get_kq(self, Q):
        F = self.stft.shape[0]
        # get KQ
        inner_dim = Q.shape[0] * Q.shape[1]
        KQ = (
                self.K.reshape(-1, inner_dim) @ Q.reshape(inner_dim, -1)
        )
        KQ = KQ.reshape(
            F, self.K.shape[1], self.K.shape[2], Q.shape[-1])
        return KQ

    def _update(self, P, Q):
        KQ = self._get_kq(Q)
        sigma = self._update_sigma(P, Q, KQ)
        P = self._update_P(P, sigma, KQ)
        sigma = self._update_sigma(P, Q, KQ)
        Q = self._update_Q(P, sigma, Q)
        return P, Q

    def run(self):
        P, Q = self.initialize_parameters()
        for i in range(self.num_iterations):
            P, Q = self._update(P, Q)

        KQ = self._get_kq(Q)
        KQ = KQ.reshape(KQ.shape[0], -1, 1, KQ.shape[-1])
        sigma_j = KQ / P[:, None, ...]
        sigma_j = sigma_j / (self.eps + sigma_j.sum(dim=-1)[..., None])
        sigma_j = self._convert_to_numpy(sigma_j)

        self.projection_set = self.projection_set.reshape(
            self.projection_set.shape[0],
            self.projection_set.shape[1] * self.projection_set.shape[2],
            self.projection_set.shape[-1]
        )

        self.inverse_projection_set = np.linalg.pinv(self.projection_set)

        cf_j = (
                (self.projection_set @
                 self.stft.transpose(0, 2, 1))[..., None]
                * sigma_j
        )
        shape = cf_j.shape

        reconstructions = (
                self.inverse_projection_set @
                cf_j.reshape(
                    cf_j.shape[0],
                    cf_j.shape[1],
                    -1
                )
        )
        reconstructions = reconstructions.reshape(
            shape[0], self.stft.shape[-1], -1, shape[-1]
        )

        self.reconstructions = np.swapaxes(reconstructions, 1, 2)

        return reconstructions

    def make_audio_signals(self):
        estimates = []
        for j in range(self.reconstructions.shape[-1]):
            estimate_stft = self.reconstructions[..., j]
            estimate = self.audio_signal.make_copy_with_stft_data(estimate_stft)
            estimate.istft()
            estimates.append(estimate)
        return estimates
