import numpy as np
import itertools
from .separation_base import SeparationBase
from ..core import utils
from ..core import constants

class MultichannelWienerFilter(SeparationBase):
    def __init__(
        self,
        input_audio_signal,
        source_signals,
        use_librosa_stft=constants.USE_LIBROSA_STFT
    ):
        """Perform multichannel wiener filtering on a mixture signal given a list of separated
        source estimates. A common post-processing step for many stereo source separation
        applications to enforce spatial stability in the separated sources. Based on
        the implementation in SIGSEP:

        https://github.com/sigsep/sigsep-mus-oracle/blob/master/MWF.py
        
        Arguments:
            input_audio_signal {AudioSignal} -- AudioSignal containing mixture
            source_signals {list} -- List of AudioSignal objects containing separated sources
        
        Keyword Arguments:
            use_librosa_stft {bool} -- Whether to use librosa stft (default: {False})
        
        Raises:
            ValueError -- [description]
            ValueError -- [description]
            ValueError -- [description]
        """

        super(MultichannelWienerFilter, self).__init__(
            input_audio_signal=input_audio_signal, 
        )

        self.sources = utils.verify_audio_signal_list_strict(source_signals)

        # Make sure input_audio_signal has the same settings as sources_list
        if self.audio_signal.num_channels < 2:
            raise ValueError('input_audio_signal must have at least 2 channels!')
        if self.audio_signal.sample_rate != self.sources[0].sample_rate:
            raise ValueError('input_audio_signal must have the same sample rate as entries of sources_list!')
        if self.audio_signal.num_channels != self.sources[0].num_channels:
            raise ValueError('input_audio_signal must have the same number of channels as entries of sources_list!')

        # Propogate stft params of mix to each signal
        for source in self.sources:
            source.stft_params = self.audio_signal.stft_params

        self.use_librosa_stft = constants.USE_LIBROSA_STFT
        self.power_spectral_density = None
        self.spatial_covariance = None
        self.mix_covariance_inv = None

    def _compute_spectrograms(self):
        """Compute spectrograms for each stft. Transposes each STFT from 
        (num_frequencies, num_time, num_channels) to (num_channels, 
        num_frequencies, num_time). Channels goes first to make indexing the
        spatial covariance matrices easier.
        """

        self.mix_stft = self.audio_signal.stft().transpose(2, 0, 1)
        self.source_stfts = [s.stft().transpose(2, 0, 1) for s in self.sources]

    def run(self):
        self._compute_spectrograms()
        power_spectral_density = []
        spatial_covariance = []
        eps = np.finfo(np.float).eps
        # I: num_channels, F: num_frequencies, T: num_time
        I, F, T = self.mix_stft.shape

        for source_stft in self.source_stfts:
            # Learn PSD and spatial covariance matrix
            # 1) compute observed covariance for source
            _spatial_covariance = np.zeros((F, T, I, I), dtype='complex')

            for (i1, i2) in itertools.product(range(I), range(I)):
                _spatial_covariance[..., i1, i2] = (
                    source_stft[i1, ...] * np.conj(source_stft[i2, ...])
                )

            # 2) compute first naive estimate of the source spectrogram as the
            #    average of spectrogram over channels

            _power_spectral_density = np.mean(np.abs(source_stft) ** 2, axis=0)

            # 3) take the spatial covariance matrix as the average of
            #    the observed Rjj weighted Rjj by 1/Pj. This is because the
            #    covariance is modeled as Pj Rj

            mean_spatial_covariance = np.mean(
                _spatial_covariance / (_power_spectral_density[..., None, None] + eps),
                axis=1
            )

            # add some regularization to this estimate: normalize and add small
            # identify matrix, so we are sure it behaves well numerically.

            mean_spatial_covariance = (
                mean_spatial_covariance * I / 
                (
                    np.trace(mean_spatial_covariance) +
                    eps * np.tile(np.eye(I, dtype='complex')[None, ...], (F, 1, 1))
                )
            )


            # 4) Now refine the power spectral density estimate. This is to better
            #    estimate the PSD in case the source has some correlations between
            #    channels.

            _spatial_covariance_inv = np.linalg.inv(mean_spatial_covariance)

            _power_spectral_density = 0
            for (i1, i2) in itertools.product(range(I), range(I)):                
                _power_spectral_density += (
                    1. / I * np.real(
                        _spatial_covariance_inv[:, i1, i2][:, None] *
                        _spatial_covariance[..., i2, i1]
                    )
                )
            
            power_spectral_density.append(_power_spectral_density)
            spatial_covariance.append(mean_spatial_covariance)

        mix_covariance = 0
        
        for i in range(len(self.sources)):
            mix_covariance += (
                power_spectral_density[i][..., None, None] * 
                spatial_covariance[i][:, None, ...]
            )
                
        self.mix_covariance_inv = np.linalg.inv(mix_covariance)
        self.power_spectral_density = power_spectral_density
        self.spatial_covariance = spatial_covariance

        return None

    def make_audio_signals(self):
        if self.mix_covariance_inv is None:
            raise ValueError('Cannot make audio signals prior to running algorithm!')
        
        # I: num_channels, F: num_frequencies, T: num_time
        I, F, T = self.mix_stft.shape
        source_signals = []
        
        # separate sources using PSD and spatial covariance matrices
        for i in range(len(self.sources)):
            multichannel_wiener_gain = np.zeros(
                self.mix_covariance_inv.shape, dtype='complex'
            )
            # Project PSD onto spatial covariance
            psd_sc = (
                self.power_spectral_density[i][..., None, None] *
                self.spatial_covariance[i][:, None, ...]
            )

            for (i1, i2, i3) in itertools.product(range(I), range(I), range(I)):
                multichannel_wiener_gain[..., i1, i2] += (
                    psd_sc[..., i1, i3] * self.mix_covariance_inv[..., i3, i2]
                )
            psd_sc = 0 # free memory?

            source_stft = 0
            for i in range(I):
                source_stft += (
                    multichannel_wiener_gain[..., i] * self.mix_stft[i, ..., None]
                )
            source_signal = self.audio_signal.make_copy_with_stft_data(source_stft)
            source_signal.stft_params = self.audio_signal.stft_params
            source_signal.istft(truncate_to_length=self.audio_signal.signal_length)
            source_signals.append(source_signal)
        
        return source_signals