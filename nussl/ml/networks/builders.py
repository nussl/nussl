"""
Functions that make it easy to build commonly used source separation architectures.
Currently contains mask inference, deep clustering, and chimera networks that are
based on recurrent neural networks. These functions are a good place to start when
creating your own network toplogies. Since there can be dependencies between layers
depending on input size, it's good to work this out in a function like those below.
"""


def build_recurrent_mask_inference(num_features, hidden_size, num_layers, bidirectional,
                                   dropout, num_sources, mask_activation, num_audio_channels=1,
                                   rnn_type='lstm', normalization_class='BatchNorm',
                                   mix_key='mix_magnitude'):
    """
    Builds a config for a mask inference network that can be passed to 
    SeparationModel. This mask inference network uses a recurrent neural network (RNN)
    to process the input representation.
    
    Args:
        num_features (int): Number of features in the input spectrogram (usually means
          window length of STFT // 2 + 1.)
        hidden_size (int): Hidden size of the RNN.
        num_layers (int): Number of layers in the RNN.
        bidirectional (int): Whether the RNN is bidirectional.
        dropout (float): Amount of dropout to be used between layers of RNN.
        num_sources (int): Number of sources to create masks for. 
        mask_activation (list of str): Activation of the mask ('sigmoid', 'softmax', etc.). 
          See ``nussl.ml.networks.modules.Embedding``. 
        num_audio_channels (int): Number of audio channels in input (e.g. mono or stereo).
          Defaults to 1.
        rnn_type (str, optional): RNN type, either 'lstm' or 'gru'. Defaults to 'lstm'.
        normalization_class (str, optional): Type of normalization to apply, either
          'InstanceNorm' or 'BatchNorm'. Defaults to 'BatchNorm'.
        mix_key (str, optional): The key to look for in the input dictionary that contains
          the mixture spectrogram. Defaults to 'mix_magnitude'.
    
    Returns:
        dict: A recurrent mask inference network configuration that can be passed to
          SeparationModel.
    """

    # define the building blocks
    modules = {
        mix_key: {},
        'log_spectrogram': {
            'class': 'AmplitudeToDB'
        },
        'normalization': {
            'class': normalization_class,
        },
        'recurrent_stack': {
            'class': 'RecurrentStack',
            'args': {
                'num_features': num_features,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'bidirectional': bidirectional,
                'dropout': dropout,
                'rnn_type': rnn_type
            }
        },
        'mask': {
            'class': 'Embedding',
            'args': {
                'num_features': num_features,
                'hidden_size': hidden_size * 2 if bidirectional else hidden_size,
                'embedding_size': num_sources,
                'activation': mask_activation,
                'num_audio_channels': num_audio_channels
            }
        },
        'estimates': {
            'class': 'Mask',
        },
    }

    # define the topology
    connections = [
        ['log_spectrogram', [mix_key, ]],
        ['normalization', ['log_spectrogram', ]],
        ['recurrent_stack', ['normalization', ]],
        ['mask', ['recurrent_stack', ]],
        ['estimates', ['mask', mix_key]]
    ]

    # define the outputs
    output = ['estimates', 'mask']

    # put it together
    config = {
        'modules': modules,
        'connections': connections,
        'output': output
    }

    return config


def build_recurrent_dpcl(num_features, hidden_size, num_layers, bidirectional,
                         dropout, embedding_size, embedding_activation, num_audio_channels=1,
                         rnn_type='lstm',
                         normalization_class='BatchNorm', mix_key='mix_magnitude'):
    """
    Builds a config for a deep clustering network that can be passed to 
    SeparationModel. This deep clustering network uses a recurrent neural network (RNN)
    to process the input representation.
    
    Args:
        num_features (int): Number of features in the input spectrogram (usually means
          window length of STFT // 2 + 1.)
        hidden_size (int): Hidden size of the RNN.
        num_layers (int): Number of layers in the RNN.
        bidirectional (int): Whether the RNN is bidirectional.
        dropout (float): Amount of dropout to be used between layers of RNN.
        embedding_size (int): Embedding dimensionality of the deep clustering network. 
        embedding_activation (list of str): Activation of the embedding ('sigmoid', 'softmax', etc.). 
          See ``nussl.ml.networks.modules.Embedding``. 
        num_audio_channels (int): Number of audio channels in input (e.g. mono or stereo).
          Defaults to 1.
        rnn_type (str, optional): RNN type, either 'lstm' or 'gru'. Defaults to 'lstm'.
        normalization_class (str, optional): Type of normalization to apply, either
          'InstanceNorm' or 'BatchNorm'. Defaults to 'BatchNorm'.
        mix_key (str, optional): The key to look for in the input dictionary that contains
          the mixture spectrogram. Defaults to 'mix_magnitude'.
    
    Returns:
        dict: A recurrent deep clustering network configuration that can be passed to
          SeparationModel.
    """

    # define the building blocks
    modules = {
        mix_key: {},
        'log_spectrogram': {
            'class': 'AmplitudeToDB'
        },
        'normalization': {
            'class': normalization_class,
        },
        'recurrent_stack': {
            'class': 'RecurrentStack',
            'args': {
                'num_features': num_features,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'bidirectional': bidirectional,
                'dropout': dropout,
                'rnn_type': rnn_type
            }
        },
        'embedding': {
            'class': 'Embedding',
            'args': {
                'num_features': num_features,
                'hidden_size': hidden_size * 2 if bidirectional else hidden_size,
                'embedding_size': embedding_size,
                'activation': embedding_activation,
                'num_audio_channels': num_audio_channels
            }
        },
    }

    # define the topology
    connections = [
        ['log_spectrogram', ['mix_magnitude', ]],
        ['normalization', ['log_spectrogram', ]],
        ['recurrent_stack', ['normalization', ]],
        ['embedding', ['recurrent_stack', ]],
    ]

    # define the outputs
    output = ['embedding']

    # put it together
    config = {
        'modules': modules,
        'connections': connections,
        'output': output
    }

    return config


def build_recurrent_chimera(num_features, hidden_size, num_layers, bidirectional,
                            dropout, embedding_size, embedding_activation, num_sources,
                            mask_activation,
                            num_audio_channels=1, rnn_type='lstm', normalization_class='BatchNorm',
                            mix_key='mix_magnitude'):
    """
    Builds a config for a Chimera network that can be passed to SeparationModel. 
    Chimera networks are so-called because they have two "heads" which can be trained
    via different loss functions. In traditional Chimera, one head is trained using a
    deep clustering loss while the other is trained with a mask inference loss. 
    This Chimera network uses a recurrent neural network (RNN) to process the input 
    representation.
    
    Args:
        num_features (int): Number of features in the input spectrogram (usually means
          window length of STFT // 2 + 1.)
        hidden_size (int): Hidden size of the RNN.
        num_layers (int): Number of layers in the RNN.
        bidirectional (int): Whether the RNN is bidirectional.
        dropout (float): Amount of dropout to be used between layers of RNN.
        embedding_size (int): Embedding dimensionality of the deep clustering network. 
        embedding_activation (list of str): Activation of the embedding ('sigmoid', 'softmax', etc.). 
          See ``nussl.ml.networks.modules.Embedding``. 
        num_sources (int): Number of sources to create masks for. 
        mask_activation (list of str): Activation of the mask ('sigmoid', 'softmax', etc.). 
          See ``nussl.ml.networks.modules.Embedding``. 
        num_audio_channels (int): Number of audio channels in input (e.g. mono or stereo).
          Defaults to 1.
        rnn_type (str, optional): RNN type, either 'lstm' or 'gru'. Defaults to 'lstm'.
          normalization_class (str, optional): Type of normalization to apply, either
        'InstanceNorm' or 'BatchNorm'. Defaults to 'BatchNorm'.
        mix_key (str, optional): The key to look for in the input dictionary that contains
          the mixture spectrogram. Defaults to 'mix_magnitude'.
    
    Returns:
        dict: A recurrent Chimera network configuration that can be passed to
          SeparationModel.
    """

    # define the building blocks
    modules = {
        mix_key: {},
        'log_spectrogram': {
            'class': 'AmplitudeToDB'
        },
        'normalization': {
            'class': normalization_class,
        },
        'recurrent_stack': {
            'class': 'RecurrentStack',
            'args': {
                'num_features': num_features,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'bidirectional': bidirectional,
                'dropout': dropout,
                'rnn_type': rnn_type
            }
        },
        'embedding': {
            'class': 'Embedding',
            'args': {
                'num_features': num_features,
                'hidden_size': hidden_size * 2 if bidirectional else hidden_size,
                'embedding_size': embedding_size,
                'activation': embedding_activation,
                'num_audio_channels': num_audio_channels
            }
        },
        'mask': {
            'class': 'Embedding',
            'args': {
                'num_features': num_features,
                'hidden_size': hidden_size * 2 if bidirectional else hidden_size,
                'embedding_size': num_sources,
                'activation': mask_activation,
                'num_audio_channels': num_audio_channels
            }
        },
        'estimates': {
            'class': 'Mask',
        },
    }

    # define the topology
    connections = [
        ['log_spectrogram', ['mix_magnitude', ]],
        ['normalization', ['log_spectrogram', ]],
        ['recurrent_stack', ['normalization', ]],
        ['embedding', ['recurrent_stack', ]],
        ['mask', ['recurrent_stack', ]],
        ['estimates', ['mask', 'mix_magnitude']]
    ]

    # define the outputs
    output = ['embedding', 'estimates', 'mask']

    # put it together
    config = {
        'modules': modules,
        'connections': connections,
        'output': output
    }

    return config
