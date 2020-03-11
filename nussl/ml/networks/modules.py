import torch
import torch.nn as nn
import librosa
import numpy as np
from ..unfold import GaussianMixtureTorch
from torch.utils.checkpoint import checkpoint
import warnings

class AmplitudeToDB(nn.Module):
    """
    Takes a magnitude spectrogram and converts it to a log
    amplitude spectrogram in decibels.
    
    Args:
        data (torch.Tensor): Magnitude spectrogram to convert to
        log spectrogram.
        ref (float): reference value. Defaults to 1.0.
        amin (float): lowest possible value for numerical stability.
        Defaults to 1e-8.
    
    Returns:
        [type]: [description]
    """
    def forward(self, data, ref=1.0, amin=1e-4):
        data = torch.abs(data) ** 2
        amin = amin ** 2
        ref = ref ** 2
        data = 10.0 * torch.log10(torch.clamp(data, min=amin))
        data -= 10.0 * np.log10(np.maximum(amin, ref))
        return data

class BatchNorm(nn.Module):
    """
    Applies a batch norm layer. Defaults to using only 1 feature, commonly
    used at the very beginning of the network to normalize the input spectrogram.

    Data comes and goes as (nb, nt, nf, nc). Inside this module, the data undergoes
    the following procedure:

    1. It is first reshaped to (nb, nf, nt, nc)
    2. Data is reshaped to (nb, nf, nt * nc).
    3. ``BatchNorm1d`` is applied with ``num_features`` to data.
    4. Data is reshaped back to (nb, nt, nf, nc) and returned.
    
    Args:
        num_features (int): num_features argument to BatchNorm1d, defaults to 1.
        **kwargs (dict): additional keyword arguments that can be passed to BatchNorm2d.
    
    Returns:
        torch.Tensor: modified input data tensor with batch norm.
    """
    def __init__(self, num_features=1, **kwargs):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.add_module('batch_norm', nn.BatchNorm1d(self.num_features, **kwargs))
    
    def forward(self, data):
        data = data.transpose(2, 1)
        shape = data.shape
        new_shape = (shape[0], self.num_features, -1)

        data = data.reshape(new_shape)
        data = self.batch_norm(data)
        data = data.reshape(shape)
        data = data.transpose(2, 1)
        return data

class InstanceNorm(nn.Module):
    """
    Applies an instance norm layer. Defaults to using only 1 feature, commonly
    used at the very beginning of the network to normalize the input spectrogram.

    Data comes and goes as (nb, nt, nf, nc). Inside this module, the data undergoes
    the following procedure:

    1. It is first reshaped to (nb, nf, nt, nc)
    2. Data is reshaped to (nb, nf, nt * nc).
    3. ``InstanceNorm1d`` is applied with ``num_features`` to data.
    4. Data is reshaped back to (nb, nt, nf, nc) and returned.
    
    Args:
        use_instance_norm (bool): Skip normalization or not.
        num_features (int): num_features argument to InstanceNorm1d, defaults to 1.
        **kwargs (dict): additional keyword arguments that are passed to InstanceNorm2d.
    
    Returns:
        torch.Tensor: modified input data tensor with instance norm applied.
    """
    def __init__(self, num_features=1, **kwargs):
        super(InstanceNorm, self).__init__()
        self.num_features = num_features
        self.add_module('instance_norm', nn.InstanceNorm1d(self.num_features, **kwargs))
    
    def forward(self, data):
        data = data.transpose(2, 1)
        shape = data.shape
        new_shape = (shape[0], self.num_features, -1)

        data = data.reshape(new_shape)
        data = self.instance_norm(data)
        data = data.reshape(shape)
        data = data.transpose(2, 1)
        return data

class MelProjection(nn.Module):
    def __init__(self, sample_rate, num_frequencies, num_mels, direction='forward', 
                clamp=False, trainable=False):
        """
        MelProjection takes as input a time-frequency representation (e.g. a spectrogram, or a mask) and outputs a mel
        project that can be learned or fixed. The initialization uses librosa to get a mel filterbank. Direction
        controls whether it is a forward transform or the inverse transform (e.g. back to spectrogram).

        Args:
            sample_rate: (int) Sample rate of audio for computing the mel filters.
            num_frequencies: (int) Number of frequency bins in input spectrogram.
            num_mels: (int) Number of mel bins in output mel spectrogram. if num_mels < 0, this does nothing
              other than clamping the output of clamp is True
            direction: (str) Which direction to go in (either 'forward' - to mel, or 'backward' - to frequencies).
              Defaults to 'forward'.
            clamp: (bool) Whether to clamp the output values of the transform between 0.0 and 1.0. Used for transforming
              a mask in and out of the mel-domain. Defaults to False.
            trainable: (bool) Whether the mel transform can be adjusted by the optimizer. Defaults to False.
        """
        super(MelProjection, self).__init__()
        self.num_mels = num_mels
        if direction not in ['backward', 'forward']:
            raise ValueError("direction must be one of ['backward', 'forward']!")
        self.direction = direction
        self.clamp = clamp

        if self.num_mels > 0:
            shape = (
                (num_frequencies, num_mels) 
                if self.direction == 'forward' else 
                (num_mels, num_frequencies)
            )
            self.add_module('transform', nn.Linear(*shape))

            mel_filters = librosa.filters.mel(sample_rate, 2 * (num_frequencies - 1), num_mels)
            mel_filters = (mel_filters.T / (mel_filters.sum(axis=1) + 1e-8)).T
            filter_bank = (
                mel_filters 
                if self.direction == 'forward' 
                else np.linalg.pinv(mel_filters)
            )

            for name, param in self.transform.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                if 'weight' in name:
                    param.data = torch.from_numpy(filter_bank).float()
                param.requires_grad_(trainable)

    def forward(self, data):
        """
        Args:
            data: Representation - shape: 
              (batch_size, sequence_length, num_frequencies or num_mels, num_sources)

        Returns:
            Mel-spectrogram or time-frequency representation of shape:
              (batch_size, sequence_length, num_mels or num_frequencies, num_sources).
        """

        if self.num_mels > 0:
            data = data.transpose(2, -1)
            data = self.transform(data)
            data = data.transpose(2, -1)

        if self.clamp:
            data = data.clamp(0.0, 1.0)
        return data.contiguous()


class Embedding(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, activation, 
                 num_audio_channels=1, dim_to_embed=-1):
        """
        Maps output from an audio representation module (e.g. RecurrentStack, 
        DilatedConvolutionalStack) to an embedding space. The output shape is 
        (batch_size, sequence_length, num_features, embedding_size). The embeddings can
        be passed through an activation function. If activation is 'softmax' or 
        'sigmoid', and embedding_size is equal to the number of sources, this module 
        can be used to implement a mask inference network (or a mask inference
        head in a Chimera network setup).

        Args:
            num_features (int): Number of features being mapped for each frame. 
              Either num_frequencies, or if used with MelProjection, num_mels if using 
              RecurrentStack. Should be 1 if using DilatedConvolutionalStack. 

            hidden_size (int): Size of output from RecurrentStack (hidden_size) or 
              DilatedConvolutionalStack (num_filters). If RecurrentStack is bidirectional, 
              this should be set to 2 * hidden_size.
            
            embedding_size (int): Dimensionality of embedding.
            
            activation (list of str): Activation functions to be applied. Options 
              are 'sigmoid', 'tanh', 'softmax', 'relu'. Unit normalization can be applied by 
              adding 'unit_norm' in list (e.g. ['sigmoid', unit_norm']).

            dim_to_embed (int): Which dimension of the input to apply the embedding to.
              Defaults to -1 (the last dimension).
        """
        super(Embedding, self).__init__()
        self.add_module(
            'linear', 
            nn.Linear(hidden_size, num_features * num_audio_channels * embedding_size)
        )
        self.num_features = num_features
        self.num_audio_channels = num_audio_channels
        self.activation = activation
        self.embedding_size = embedding_size
        self.dim_to_embed = dim_to_embed

        for name, param in self.linear.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, data):
        """
        Args:
            data: output from RecurrentStack or ConvolutionalStack. Shape is:
              (num_batch, ..., hidden_size or num_filters)

        Returns:
            An embedding (with an optional activation) for each point in the 
              representation of shape (num_batch, ..., embedding_size).
        """
        shape = data.shape
        data = data.transpose(self.dim_to_embed, -1)
        data = self.linear(data)
        data = data.transpose(-1, self.dim_to_embed)

        shape = shape[:-1] + (
            self.num_features, self.num_audio_channels, self.embedding_size,)
        data = data.reshape(shape)

        if 'sigmoid' in self.activation:
            data = torch.sigmoid(data)
        elif 'tanh' in self.activation:
            data = torch.tanh(data)
        elif 'relu' in self.activation:
            data = torch.relu(data)
        elif 'softmax' in self.activation:
            data = torch.softmax(data, dim=-1)

        if 'unit_norm' in self.activation:
            data = nn.functional.normalize(data, dim=-1, p=2)

        return data

class Mask(nn.Module):
    """
    Takes a mask and applies it to a representation. Mask and representation must match
    in their first 3 dimensions (nb, nt, nf). The last
    dimension of the representation is unsqueezed to match the mask shape. So if there
    are ``ns`` sources to separate, the mask shape will be (nb, nt, nf, ns), the 
    representation shape will be (nb, nt, nf).

    Representation gets unsqueezed to (nb, nt, nf, 1). Multiplying with the mask
    broadcasts, resulting in (nb, nt, nf, ns) output corresponding to each separated
    source from the representation.
    """
    def __init__(self):
        super(Mask, self).__init__()

    def forward(self, mask, representation):
        # add a source dimension
        representation = representation.unsqueeze(-1).expand_as(mask)
        return mask * representation

class RecurrentStack(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, bidirectional, dropout, 
                 rnn_type='lstm'):
        """
        Creates a stack of RNNs used to process an audio sequence represented as 
        (sequence_length, num_features). With bidirectional = True, hidden_size = 600, 
        num_layers = 4, rnn_type='lstm', and dropout = .3, this becomes the 
        audio processor used in deep clustering networks, deep attractor networks, etc. 
        Note that batch_first is set to True here.

        Args:
            num_features: (int) Number of features being mapped for each frame. 
              Either num_frequencies, or if used with MelProjection, num_mels.
            hidden_size: (int) Hidden size of recurrent stack for each layer.
            num_layers: (int) Number of layers in stack.
            bidirectional: (int) True makes this a BiLSTM or a BiGRU. Note that this 
              doubles the hidden size.
            dropout: (float) Dropout between layers.
            rnn_type: (str) LSTM ('lstm') or GRU ('gru').
        """
        super(RecurrentStack, self).__init__()
        if rnn_type not in ['lstm', 'gru']:
            raise ValueError("rnn_type must be one of ['lstm', 'gru']!")

        RNNClass = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.add_module(
            'rnn', RNNClass(
                num_features, hidden_size, num_layers, batch_first=True,
                bidirectional=bidirectional, dropout=dropout))

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        for names in self.rnn._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)

    def forward(self, data):
        """
        Args:
            data: Audio representation to be processed. Should be of shape:
            (num_batch, sequence_length, ...).

        Returns:
            Outputs the features after processing of the RNN. Shape is:
            (num_batch, sequence_length, hidden_size or hidden_size*2 if 
            bidirectional=True)
        """
        shape = data.shape
        data = data.reshape(shape[0], shape[1], -1)
        self.rnn.flatten_parameters()
        data = self.rnn(data)[0]
        return data

class ConvolutionalStack2D(nn.Module):   
    def __init__(self, in_channels, channels, dilations, filter_shapes, residuals, 
        batch_norm=True, use_checkpointing=False):
        """Implements a stack of dilated convolutional layers for source separation from 
        the following papers:

            Mobin, Shariq, Brian Cheung, and Bruno Olshausen. "Convolutional vs. recurrent 
            neural networks for audio source separation." 
            arXiv preprint arXiv:1803.08629 (2018). https://arxiv.org/pdf/1803.08629.pdf

            Yu, Fisher, Vladlen Koltun, and Thomas Funkhouser. "Dilated residual networks." 
            Proceedings of the IEEE conference on computer vision and pattern recognition. 
            2017. https://arxiv.org/abs/1705.09914
        
        Args:
            in_channels (int): Number of channels in input
            channels (list of int): Number of channels for each layer
            dilations (list of ints or int tuples): Dilation rate for each layer. If 
              int, it is same in both height and width. If tuple, tuple is defined as
              (height, width). 
            filter_shapes (list of ints or int tuples): Filter shape for each layer. If 
              int, it is same in both height and width. If tuple, tuple is defined as
              (height, width).
            residuals (list of bool): Whether or not to keep a residual connection at
              each layer.
            batch_norm (bool): Whether to use BatchNorm or not at each layer (default: True)
            use_checkpointing (bool): Whether to use torch's checkpointing functionality 
              to reduce memory usage.
        
        Raises:
            ValueError -- All the input lists must be the same length.
        """
        for x in [dilations, filter_shapes, residuals]:
            if len(x) != len(channels):
                raise ValueError(
                    f"All lists (channels, dilations, filters, residuals) should have" 
                    f"the same length!"
                )
        super().__init__()

        if any([d != 1 for d in dilations]):
            warnings.warn(
                "You specified a dilation != 1. Input size and output size are " 
                "not guaranteed to be the same! This is due to the lacking of "
                "padding = 'same' in PyTorch."
            )

        self.dilations = dilations
        self.filter_shapes = filter_shapes
        self.residuals = residuals
        self.batch_norm = batch_norm
        self.padding = None
        self.channels = channels
        self.in_channels = in_channels
        self.use_checkpointing = use_checkpointing
        self.num_layers = len(channels)
        self.layers = nn.ModuleList()

        for i in range(len(channels)):
            self.layers.append(self._make_layer(i))

    def _make_layer(self, i):
        convolution = nn.Conv2d(
            in_channels=self.channels[i-1] if i > 0 else self.in_channels,
            out_channels=self.channels[i],
            kernel_size=self.filter_shapes[i],
            dilation=self.dilations[i],
            padding=self.filter_shapes[i] // 2,
        )

        if i == len(self.channels) - 1:
            layer = convolution
            self.add_module(f'layer{i}', layer)
            return layer

        layer = nn.Sequential()
        layer.add_module('conv', convolution)
        if self.batch_norm:
            batch_norm = nn.BatchNorm2d(self.channels[i])
            layer.add_module('batch_norm', batch_norm)
        layer.add_module('relu', nn.ReLU())
        return layer

    def layer_function(self, data, layer, previous_layer, i):
        conv_layer = layer.conv if hasattr(layer, 'conv') else layer
        if self.use_checkpointing:
            data = checkpoint(layer, (data))
        else:
            data = layer(data)
        if self.residuals[i]:
            if previous_layer is not None:
                data += previous_layer
            previous_layer = data
        return data, previous_layer

    def forward(self, data):
        """ 
        Data comes in as: [num_batch, sequence_length, num_frequencies, num_audio_channels]
        We reshape it in the forward pass so that everything works to:
          [num_batch, num_audio_channels, sequence_length, num_frequencies]
        After this input is processed, the shape is then:
          [num_batch, num_output_channels, sequence_length, num_frequencies]
        We transpose again to make the shape:
          [num_batch, sequence_length, num_frequencies, num_output_channels]
        So it can be passed to an Embedding module.
        """
        data =  data.permute(0, 3, 1, 2)
        previous_layer = None
        for i in range(self.num_layers):
            data, previous_layer = self.layer_function(
                data, self.layers[i], previous_layer, i
            )
        data = data.permute(0, 2, 3, 1)
        return data
