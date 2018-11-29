import torch
import torch.nn as nn
import librosa
import numpy as np
from .clustering import GMM


class MelProjection(nn.Module):
    def __init__(self, sample_rate, num_frequencies, num_mels, direction='forward', clamp=False, trainable=False):
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
            shape = (num_frequencies, num_mels) if self.direction == 'forward' else (num_mels, num_frequencies)
            self.add_module('transform', nn.Linear(*shape))

            mel_filters = librosa.filters.mel(sample_rate, 2 * (num_frequencies - 1), num_mels)
            mel_filters = (mel_filters.T / mel_filters.sum(axis=1)).T
            filter_bank = mel_filters if self.direction == 'forward' else np.linalg.pinv(mel_filters)

            for name, param in self.transform.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                if 'weight' in name:
                    param.data = torch.from_numpy(filter_bank).float()
                param.requires_grad_(trainable)

    def forward(self, data):
        """
        Args:
            data: Representation - shape: (batch_size, sequence_length, num_frequencies or num_mels, num_sources)

        Returns:
            Mel-spectrogram or time-frequency representation of shape:
                (batch_size, sequence_length, num_mels or num_frequencies, num_sources). num_sources squeezed if 1.
        """

        if self.num_mels > 0:
            if len(data.shape) < 4:
                data = data.unsqueeze(-1)

            data = data.permute(0, 1, 3, 2)
            data = self.transform(data)
            data = data.permute(0, 1, 3, 2)

            if data.shape[-1] == 1:
                data = data.squeeze(-1)

        if self.clamp:
            data = data.clamp(0.0, 1.0)
        return data


class Embedding(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, activation):
        """
        Maps output from an audio representation function (e.g. RecurrentStack, ConvolutionalStack) to an embedding
        space. The output shape is (batch_size, sequence_length, num_features, embedding_size). The embeddings can
        be passed through an activation function. If activation is 'softmax' or 'sigmoid', and embedding_size is equal
        to the number of sources, this module can be used to implement a mask inference network (or a mask inference
        head in a Chimera network setup).

        Args:
            num_features: (int) Number of features being mapped for each frame. Either num_frequencies, or if used with
                MelProjection, num_mels.
            hidden_size: (int) Size of output from RecurrentStack (hidden_size) or ConvolutionalStack (num_filters). If
                RecurrentStack is bidirectional, this should be set to 2 * hidden_size.
            embedding_size: (int) Dimensionality of embedding.
            activation: (str) Activation functions to be applied, separated by ':'. Options are 'sigmoid', 'tanh', 'softmax'.
                Unit normalization can be applied by adding ':unit_norm' (e.g. 'sigmoid:unit_norm'). Defaults to no
                activation.
        """
        super(Embedding, self).__init__()
        self.add_module('linear', nn.Linear(hidden_size, num_features * embedding_size))
        self.num_features = num_features
        self.activation = activation
        self.embedding_size = embedding_size

    def forward(self, data):
        """
        Args:
            data: output from RecurrentStack or ConvolutionalStack. Shape is:
                (num_batch, sequence_length, hidden_size or num_filters)
        Returns:
            An embedding (with an optional activation) for each point in the representation of shape
            (num_batch, sequence_length, num_features, embedding_size).
        """
        data = self.linear(data)

        if 'sigmoid' in self.activation:
            data = torch.sigmoid(data)
        elif 'tanh' in self.activation:
            data = torch.tanh(data)
        elif 'softmax' in self.activation:
            data = nn.functional.softmax(data, dim=-1)

        data = data.view(data.shape[0], -1, self.num_features, self.embedding_size)

        if 'unit_norm' in self.activation:
            data = nn.functional.normalize(data, dim=-1, p=2)

        return data

class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()

    def forward(self, mask, magnitude_spectrogram):
        return mask * magnitude_spectrogram.unsqueeze(-1)

class RecurrentStack(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, bidirectional, dropout, rnn_type='lstm'):
        """
        Creates a stack of RNNs used to process an audio sequence represented as (sequence_length, num_features). With
        bidirectional = True, hidden_size = 600, num_layers = 4, rnn_type='lstm', and dropout = .3, this becomes
        the state-of-the-art audio processor used in deep clustering networks, deep attractor networks, etc. Note that
        batch_first is set to True here.

        Args:
            num_features: (int) Number of features being mapped for each frame. Either num_frequencies, or if used with
                MelProjection, num_mels.
            hidden_size: (int) Hidden size of recurrent stack for each layer.
            num_layers: (int) Number of layers in stack.
            bidirectional: (int) True makes this a BiLSTM or a BiGRU. Note that this doubles the hidden size.
            dropout: (float) Dropout between layers.
            rnn_type: (str) LSTM ('lstm') or GRU ('gru').
        """
        super(RecurrentStack, self).__init__()
        if rnn_type not in ['lstm', 'gru']:
            raise ValueError("rnn_type must be one of ['lstm', 'gru']!")

        if rnn_type == 'lstm':
            self.add_module('rnn', nn.LSTM(num_features,
                                           hidden_size,
                                           num_layers,
                                           batch_first=True,
                                           bidirectional=bidirectional,
                                           dropout=dropout))
        elif rnn_type == 'gru':
            self.add_module('rnn', nn.GRU(num_features,
                                          hidden_size,
                                          num_layers,
                                          batch_first=True,
                                          bidirectional=bidirectional,
                                          dropout=dropout))

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
                (num_batch, sequence_length, num_features).

        Returns:
            Outputs the features after processing of the RNN. Shape is:
                (num_batch, sequence_length, hidden_size or hidden_size*2 if bidirectional=True)
        """
        data = self.rnn(data)[0]
        return data


class ConvolutionalStack(nn.Module):
    def __init__(self, num_filters, num_layers):
        super(ConvolutionalStack, self).__init__()
        raise NotImplementedError("Not implemented yet")

    def forward(self, data):
        return


class Clusterer(nn.Module):
    def __init__(self, n_iterations, num_clusters, covariance_type='tied:spherical', covariance_init=1.0):
        """

        Args:
            clustering_type:
            n_iterations:
            num_clusters:
            covariance_type:
            covariance_init:
        """
        super(Clusterer, self).__init__()

        self.add_module('clusterer', GMM(n_clusters=num_clusters,
                                         n_iterations=n_iterations,
                                         covariance_type=covariance_type,
                                         covariance_init=covariance_init))

    def forward(self, data, parameters=None):
        num_batch, sequence_length, num_features, embedding_size = data.shape
        data = data.reshape(num_batch, sequence_length*num_features, embedding_size)
        output = self.clusterer(data, parameters)

        return


class CentroidGenerator(nn.Module):
    def __init__(self, embedding_size, centroid_type, activation, covariance_type):
        super(CentroidGenerator, self).__init__()
        return

    def forward(self, data):
        return