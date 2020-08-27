import nussl
from torch import nn
import torch
from .... import AudioSignal

class FilterBank(nn.Module):
    """
    Base class for implementing short-time filter-bank style transformations
    of an audio signal. 
    
    This class accepts two different tensors, as there are two modes it can
    be called in:

    - transform: This takes an audio signal and maps it to a spectral 
      representation by applying the internal filterbank.

    - inverse: This takes a spectral representation and maps it back to
      the audio domain.

    There are two unimplemented functions in this class that must be
    implemented by the subclass:

    - ``get_transform_filters``: This should produce a filter bank that can be
      applied to the audio signal, of shape ``(filter_length, num_filters)``. The
      filter bank is applied by chunking the signal into overlapping segments
      using ``nn.Unfold``, and performing a matrix multiplication of the chunks 
      (after permuting the dimensions appropriately) with the filter bank.

    - ``get_inverse_filters``: This should produce a filter bank that maps the
      spectral representation back to the audio domain. This is done by first
      permuting the dimensions of the spectral representations appropriately,
      then matrix multiplying the spectral representation with the filter bank.
      Finally, the signal is resynthesized via overlap-add using ``nn.Fold``.

    Windowing is applied to the signal, according to ``window_type``, which
    can be any of the windows found in ``nussl.core.constants.ALL_WINDOWS``.

    This can also be applied to multiple sources at once, if they are on the
    last axis after all of the feature dimensions. If the number of features
    in the data is greater than the expected number by one, then the last
    dimension is assumed to be the source dimension. This dimension is then
    mapped and merged with the batch dimension so the transforms can be applied
    to all of the sources. Before returning the data, the source dimension is 
    moved back to the last dimension.

    In the forward pass, keyword arguments can also be passed through. These
    keyword arguments get passed through to ``apply_filter`` and ``get_filters``,
    in case these should be conditioned on something during the forward pass.

    Note:
        The output dimensionality may not always match what is given by 
        something like scipy.stft (e.g. might be off by one in frames) for 
        some hop lengths! However, an attempt was made and for hop lengths
        that are half the filter length or a quarter of the filter length, the
        number of segments will match that of scipy.stft.
    
    Args:
        num_filters (int): Number of filters in both filterbanks.
        filter_length (int, optional): Length of each filter. Defaults to None,
            equal to num_filters.
        hop_length (int, optional): Hop length between each filter. 
            Defaults to None, half of filter_length.
        window_type (str, optional): Type of window to use. Defaults to 
            'sqrt_hann'.
        dilation (int, optional): Dilation of nn.Unfold and nn.Fold. Could be
            useful for implementing dilated convolutional frontends. 
            Defaults to 1.
        direction (str, optional): In which direction to take the input data. 
            Either 'transform' or 'inverse'. Can also be set during the 
            forward pass. Defaults to 'transform'.
        requires_grad (bool, optional): Whether to make the filterbank learnable
            during backprop. Defaults to False.
    """
    def __init__(self, num_filters, filter_length=None, hop_length=None,
                 window_type='sqrt_hann', dilation=1, direction='transform',
                 requires_grad=False):
        super().__init__()
        
        self.num_filters = num_filters
        self.filter_length = (
            num_filters 
            if filter_length is None
            else filter_length
        )
        self.hop_length = (
            self.filter_length // 2 
            if hop_length is None 
            else hop_length
        )
        
        self.direction = direction
        self.requires_grad = requires_grad
        self.dilation = dilation
        self.window_type = window_type
        self.output_length = None

        self.register_buffer('window', self._get_window())

        self.transform_filters = self.get_transform_filters()
        self.inverse_filters = self.get_inverse_filters()
        
    def _get_window(self):
        window = AudioSignal.get_window(
            self.window_type, self.filter_length
        )
        window = torch.from_numpy(window).float()
        return window.reshape(1, 1, -1, 1)
        
    def get_transform_filters(self):
        raise NotImplementedError()
        
    def get_inverse_filters(self):
        raise NotImplementedError()
    
    def get_filters(self):
        filters = (
            self.transform_filters
            if self.direction == 'transform'
            else self.inverse_filters
        )
        return filters
    
    def apply_filter(self, data, **kwargs):
        filters = self.get_filters(**kwargs)        
        data = data.transpose(-1, 2)
        data = data @ filters
        data = data.transpose(-1, 2)
        return data   
    
    def transform(self, data, **kwargs):
        ndim = data.ndim
        if ndim > 3:
            # move sources to the batch dimension
            # then fix it later
            num_sources = data.shape[-1]
            data = data.permute(0, -1, 1, 2)
            data = data.reshape(-1, *data.shape[2:])

        self.original_length = data.shape[-1]
        pad_extra = (
            (-(data.shape[-1] - self.filter_length) % self.hop_length)
            % self.filter_length
        )
        pad_tuple = (
            self.filter_length // 2,
            self.filter_length // 2 + pad_extra
        )
        data = nn.functional.pad(data, pad_tuple)
        self.output_length = data.shape[-1]
        
        num_batch, num_audio_channels, num_samples = data.shape

        unfold = nn.Unfold(
            kernel_size=(1, self.filter_length), 
            stride=(1, self.hop_length),
            dilation=self.dilation,
            padding=(0, 0)
        )
        data = data.reshape(
            num_batch * num_audio_channels,
            1, 1, num_samples
        )
        data = unfold(data)
        data = data.view(
            num_batch, num_audio_channels,
            self.filter_length, -1
        )
        data = data.permute(0, 3, 2, 1)
        data = data * self.window
        data = self.apply_filter(data, **kwargs)

        if ndim > 3:
            # then we moved sources to the batch dimension
            # we need to move it back before returning
            data = data.reshape(
                -1, num_sources, *data.shape[1:])
            data = data.permute(0, 2, 3, 4, 1)
        return data
    
    def inverse(self, data, **kwargs):
        ndim = data.ndim
        if ndim > 4:
            # move sources to the batch dimension
            # then fix it later
            num_sources = data.shape[-1]
            data = data.permute(0, -1, 1, 2, 3)
            data = data.reshape(-1, *data.shape[2:])
        
        data = self.apply_filter(data, **kwargs)

        data *= self.window.sum()
        data = data * self.window
        
        num_batch, sequence_length, num_features, num_audio_channels = (
            data.shape
        )
        
        data = data.permute(0, 3, 2, 1)
        data = data.reshape(-1, data.shape[2], data.shape[3])

        fold = nn.Fold(
            (1, self.output_length),
            kernel_size=(1, self.filter_length),
            stride=(1, self.hop_length),
            dilation=self.dilation,
            padding=(0, 0),
        )
        norm = data.new_ones(data.shape)
        norm *= self.window.view(1, -1, 1) ** 2
        
        data = fold(data)
        norm = fold(norm)
        norm[norm < 1e-10] = 1
        data = data / norm

        data = data.reshape(num_batch, num_audio_channels, -1)
        
        boundary = self.filter_length // 2
        data = data[..., boundary:-boundary]
        data = data[..., :self.original_length]

        if ndim > 4:
            # then we moved sources to the batch dimension
            # we need to move it back before returning
            data = data.reshape(
                -1, num_sources, num_audio_channels, data.shape[-1])
            data = data.permute(0, 2, 3, 1)

        return data
        
    def forward(self, data, direction=None, **kwargs):
        if direction is not None:
            self.direction = direction
        if self.direction == 'transform':
            func = self.transform
        elif self.direction == 'inverse':
            func = self.inverse
        return func(data, **kwargs)

class STFT(FilterBank):
    """
    An implementation of STFT and iSTFT using nn.Unfold, nn.Fold, and matrix
    multiplication with a Fourier basis.

    The usual way to compute an STFT is to split the signal into overlapping
    chunks, multiply each chunk with a window function, and then 
    apply an FFT to each chunk individually. Here, instead of taking the 
    FFT with something like ``torch.fft``, we instead use the matrix
    formulation of FFT. 

    To resynthesize the signal, we use inverse windowing and the pseudoinverse
    of the FFT matrix as our filterbank. We then use overlap/add and divide by
    a normalization factor to reconstruct the signal. The usual error for this
    implementation is on the order of 1e-7.

    In the `transform` direction, this class returns the magnitude and phase
    of the STFT, concatenated along a single dimension. Generally, the first
    half of this is what you would operate on (the magnitudes), while the
    second half you would keep around for reconstructing the signal later on.
    """
    def apply_filter(self, data):
        if self.direction == 'transform':
            scale = torch.sqrt(1.0 / (self.window.sum() ** 2))
            data = super().apply_filter(data)
            data *= scale
            
            eps = 1e-8
            cutoff = 1 + self.filter_length // 2
            real_part = data[..., :cutoff, :]
            imag_part = data[..., cutoff:, :]
            real_part[real_part.abs() <= eps] = eps
            imag_part[imag_part.abs() <= eps] = eps
            
            magnitude = torch.sqrt(
                real_part ** 2 + imag_part ** 2)
            phase = torch.atan2(imag_part, real_part)      
            data = torch.cat([magnitude, phase], dim=2)

        elif self.direction == 'inverse':
            cutoff = 1 + self.filter_length // 2
            magnitude = data[..., :cutoff, :]
            phase = data[..., cutoff:, :]
            data = torch.cat(
                [
                    magnitude * torch.cos(phase), 
                    magnitude * torch.sin(phase)
                ],
                dim=2
            )
            data = super().apply_filter(data)
            
        return data
    
    def _get_fft_basis(self):
        fourier_basis = torch.rfft(
            torch.eye(self.filter_length), 
            1, onesided=True
        )
        cutoff = 1 + self.filter_length // 2
        fourier_basis = torch.cat([
            fourier_basis[:, :cutoff, 0],
            fourier_basis[:, :cutoff, 1]
        ], dim=1)
        return fourier_basis.float()
    
    def get_transform_filters(self):
        fourier_basis = self._get_fft_basis()
        return nn.Parameter(fourier_basis, requires_grad=self.requires_grad)
    
    def get_inverse_filters(self):
        fourier_basis = self._get_fft_basis()
        inverse_filters = torch.pinverse(
            fourier_basis.unsqueeze(0)).squeeze(0)
        return nn.Parameter(inverse_filters, requires_grad=self.requires_grad)

class LearnedFilterBank(FilterBank):
    """
    Implements a simple learnable filter bank. The filter bank is completely
    random on initialization and should be learned during backprop. 
    `requires_grad` is set to True regardless of what the parent class
    init has.
    """
    def get_transform_filters(self):
        basis = nn.Parameter(
            torch.ones(self.filter_length, self.num_filters),
            requires_grad=True
        )
        nn.init.xavier_normal_(basis)
        return basis

    def get_inverse_filters(self):
        basis = nn.Parameter(
            torch.ones(self.num_filters, self.filter_length),
            requires_grad=True
        )
        nn.init.xavier_normal_(basis)
        return basis
