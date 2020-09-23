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
                                   normalization_args=None, mix_key='mix_magnitude'):
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
        normalization_args (dict, optional): Args to normalization class, optional.
        mix_key (str, optional): The key to look for in the input dictionary that contains
          the mixture spectrogram. Defaults to 'mix_magnitude'.
    
    Returns:
        dict: A recurrent mask inference network configuration that can be passed to
          SeparationModel.
    """
    normalization_args = {} if normalization_args is None else normalization_args

    # define the building blocks
    modules = {
        mix_key: {},
        'log_spectrogram': {
            'class': 'AmplitudeToDB',
        },
        'normalization': {
            'class': normalization_class,
            'args': normalization_args,
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
        'name': 'MaskInference',
        'modules': modules,
        'connections': connections,
        'output': output
    }

    return config


def build_recurrent_dpcl(num_features, hidden_size, num_layers, bidirectional,
                         dropout, embedding_size, embedding_activation, num_audio_channels=1,
                         rnn_type='lstm', normalization_class='BatchNorm', 
                         normalization_args=None, mix_key='mix_magnitude'):
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
        normalization_args (dict, optional): Args to normalization class, optional.
        mix_key (str, optional): The key to look for in the input dictionary that contains
          the mixture spectrogram. Defaults to 'mix_magnitude'.
    
    Returns:
        dict: A recurrent deep clustering network configuration that can be passed to
          SeparationModel.
    """
    normalization_args = {} if normalization_args is None else normalization_args

    # define the building blocks
    modules = {
        mix_key: {},
        'log_spectrogram': {
            'class': 'AmplitudeToDB',
        },
        'normalization': {
            'class': normalization_class,
            'args': normalization_args,
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
        'name': 'DeepClustering',
        'modules': modules,
        'connections': connections,
        'output': output
    }

    return config


def build_recurrent_chimera(num_features, hidden_size, num_layers, bidirectional,
                            dropout, embedding_size, embedding_activation, num_sources,
                            mask_activation, num_audio_channels=1, 
                            rnn_type='lstm', normalization_class='BatchNorm',
                            normalization_args=None, mix_key='mix_magnitude'):
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
        normalization_args (dict, optional): Args to normalization class, optional.
        mix_key (str, optional): The key to look for in the input dictionary that contains
          the mixture spectrogram. Defaults to 'mix_magnitude'.
    
    Returns:
        dict: A recurrent Chimera network configuration that can be passed to
          SeparationModel.
    """
    normalization_args = {} if normalization_args is None else normalization_args

    # define the building blocks
    modules = {
        mix_key: {},
        'log_spectrogram': {
            'class': 'AmplitudeToDB',
        },
        'normalization': {
            'class': normalization_class,
            'args': normalization_args,
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
        'name': 'Chimera',
        'modules': modules,
        'connections': connections,
        'output': output
    }

    return config

def build_open_unmix_like(num_features, hidden_size, num_layers, 
                          bidirectional, dropout, num_sources,
                          num_audio_channels=1, add_embedding=False,
                          embedding_size=20, embedding_activation='sigmoid',
                          rnn_type='lstm', mix_key='mix_magnitude'):
    """
    This is a builder for an open-unmix LIKE (UMX) architecture for music source
    separation. 
    
    The architecture is not exactly the same but is very similar for the
    most part. This architecture also has the option of having an embedding space
    attached to it, making it a UMX + Chimera network that you can regularize with
    a deep clustering loss.
    
    Args:
        num_features (int): Number of features in the input spectrogram (usually means
          window length of STFT // 2 + 1.)
        hidden_size (int): Hidden size of the RNN. Will be hidden_size // 2 if bidirectional is True.
        num_layers (int): Number of layers in the RNN.
        bidirectional (int): Whether the RNN is bidirectional.
        dropout (float): Amount of dropout to be used between layers of RNN.
        num_sources (int): Number of sources to create masks for.
        num_audio_channels (int): Number of audio channels in input (e.g. mono or stereo).
          Defaults to 1.
        add_embedding (bool): Whether or not to add an embedding layer to this to make this a 
          Chimera network. If True, then ``embedding_size`` and ``embedding_activation`` will
          be used to define this.
        embedding_size (int): Embedding dimensionality of the deep clustering network. 
        embedding_activation (list of str): Activation of the embedding ('sigmoid', 'softmax', etc.). 
          See ``nussl.ml.networks.modules.Embedding``.
        rnn_type (str, optional): RNN type, either 'lstm' or 'gru'. Defaults to 'lstm'.
        mix_key (str, optional): The key to look for in the input dictionary that contains
          the mixture spectrogram. Defaults to 'mix_magnitude'.
    
    Returns:
        dict: An OpenUnmix-like configuration that can be passed to
          SeparationModel.
    """
    # define the building blocks
    modules = {
        mix_key: {},
        'projection': {
            'class': 'Embedding',
            'args': {
                'num_features': hidden_size,
                'hidden_size': num_features * num_audio_channels,
                'embedding_size': 1,
                'activation': 'none',
                'num_audio_channels': 1,
                'bias': False,
                'reshape': False,
                'dim_to_embed': [2, 3]
            }
        },
        'recurrent_stack': {
            'class': 'RecurrentStack',
            'args': {
                'num_features': hidden_size,
                'hidden_size': (
                    hidden_size // 2 
                    if bidirectional 
                    else hidden_size
                ),
                'num_layers': num_layers,
                'bidirectional': bidirectional,
                'dropout': dropout,
                'rnn_type': rnn_type
            }
        },
        'skip_connection': {
            'class': 'Concatenate',
            'args': {
                'dim': -1
            }
        },
        'input_scalar': {
            'class': 'BatchNorm'
        },
        'output_scalar': {
            'class': 'BatchNorm'
        },
        'bn1': {
            'class': 'BatchNorm',
            'args': {
                'num_features': hidden_size
            }
        },
        'bn2': {
            'class': 'BatchNorm',
            'args': {
                'num_features': hidden_size
            }
        },
        'bn3': {
            'class': 'BatchNorm',
            'args': {
                'num_features': num_features
            }
        },
        'tanh_before_lstm': {
            'class': 'Tanh' 
        },
        'mask': {
            'class': 'ReLU'
        },
        'dense_after_lstm': {
            'class': 'Linear',
            'args': {
                'in_features': 2 * hidden_size,
                'out_features': hidden_size,
                'bias': False
            }
        },
        'inverse_projection': {
            'class': 'Embedding',
            'args': {
                'num_features': num_features,
                'hidden_size': hidden_size,
                'embedding_size': num_sources,
                'activation': 'none',
                'num_audio_channels': num_audio_channels,
                'bias': False,
                'reshape': True,
                'dim_to_embed': -1
            }
        },
        'estimates': {
            'class': 'Mask',
        }
    }

    # define the topology
    connections = [
        ['input_scalar', [mix_key]],
        ['projection', ['input_scalar']],
        ['bn1', ['projection']],
        ['tanh_before_lstm', ['bn1']],
        ['recurrent_stack', ['tanh_before_lstm']],
        ['skip_connection', ['recurrent_stack', 'tanh_before_lstm']],
        ['dense_after_lstm', ['skip_connection']],
        ['bn2', ['dense_after_lstm']],
        ['inverse_projection', ['bn2']],
        ['bn3', ['inverse_projection']],
        ['output_scalar', ['bn3']],
        ['mask', ['output_scalar']],
        ['estimates', ['mask', mix_key]]
    ]
    
    # define the outputs
    output = ['estimates', 'mask']
    
    if add_embedding:
        modules['embedding'] = {
            'class': 'Embedding',
            'args': {
                'num_features': num_features,
                'hidden_size': hidden_size,
                'embedding_size': embedding_size,
                'activation': embedding_activation,
                'num_audio_channels': num_audio_channels,
                'bias': True,
                'reshape': True,
                'dim_to_embed': -1
            }
        }
        connections.append(['embedding', ['bn2']])
        output.append('embedding')

    # put it together
    config = {
        'name': 'OpenUnmix',
        'modules': modules,
        'connections': connections,
        'output': output
    }

    return config


def build_recurrent_end_to_end(num_filters, filter_length, hop_length, window_type, 
                               hidden_size, num_layers, bidirectional, dropout, 
                               num_sources, mask_activation, num_audio_channels=1,
                               mask_complex=False, trainable=False, rnn_type='lstm', 
                               mix_key='mix_audio', normalization_class='BatchNorm',
                               normalization_args=None):
    """
    Builds a config for a BLSTM-based network that operates on the time-series. 
    Uses an STFT within the network and can apply the mixture phase to
    the estimate, or can learn a mask on the phase as well as the magnitude.
    
    Args:
        num_filters (int): Number of learnable filters in the front end network.
        filter_length (int): Length of the filters.
        hop_length (int): Hop length between frames.
        window_type (str): Type of windowing function on apply to each frame.
        hidden_size (int): Hidden size of the RNN.
        num_layers (int): Number of layers in the RNN.
        bidirectional (int): Whether the RNN is bidirectional.
        dropout (float): Amount of dropout to be used between layers of RNN.
        num_sources (int): Number of sources to create masks for. 
        mask_activation (list of str): Activation of the mask ('sigmoid', 'softmax', etc.). 
          See ``nussl.ml.networks.modules.Embedding``.
        num_audio_channels (int): Number of audio channels in input (e.g. mono or stereo).
          Defaults to 1.
        mask_complex (bool, optional): Whether to also place a mask on the complex part, or
          whether to just use the mixture phase.
        trainable (bool, optional): Whether to learn the filters, which start from a 
          Fourier basis.
        rnn_type (str, optional): RNN type, either 'lstm' or 'gru'. Defaults to 'lstm'.
        normalization_class (str, optional): Type of normalization to apply, either
          'InstanceNorm' or 'BatchNorm'. Defaults to 'BatchNorm'.
        normalization_args (dict, optional): Args to normalization class, optional.
        mix_key (str, optional): The key to look for in the input dictionary that contains
          the mixture spectrogram. Defaults to 'mix_magnitude'.
    
    Returns:
        dict: A recurrent end-to-end network configuration that can be passed to
          SeparationModel.
    """
    normalization_args = {} if normalization_args is None else normalization_args
    cutoff = num_filters // 2 + 1
    num_features = 2 * cutoff if mask_complex else cutoff

    # define the building blocks
    modules = {
        mix_key: {},
        'audio': {
            'class': 'STFT',
            'args': {
                'num_filters': num_filters,
                'filter_length': filter_length,
                'hop_length': hop_length,
                'window_type': window_type,
                'requires_grad': trainable
            }
        },
        'log_spectrogram': {
            'class': 'AmplitudeToDB'
        },
        'normalization': {
            'class': normalization_class,
            'args': normalization_args,
        },
        'split': {
            'class': 'Split',
            'args': {
                'split_sizes': (cutoff, cutoff),
                'dim': 2
            }
        },
        'concatenate': {
            'class': 'Concatenate',
            'args': {
                'dim': 2
            }
        },
        'expand': {
            'class': 'Expand',
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
    if not mask_complex:
        connections = [
            ['audio', [mix_key, {'direction': 'transform'}]],
            ['split', ['audio',]],
            ['log_spectrogram', ['split:0', ]],
            ['normalization', ['log_spectrogram']],
            ['recurrent_stack', ['normalization', ]],
            ['mask', ['recurrent_stack', ]],
            ['estimates', ['mask', 'split:0']],
            ['expand', ['mask', 'split:1']],
            ['concatenate', ['estimates', 'expand']],
            ['audio', ['concatenate', {'direction': 'inverse'}]]
        ]
    else:
        connections = [
            ['audio', [mix_key, {'direction': 'transform'}]],
            ['normalization', ['audio']],
            ['recurrent_stack', ['normalization', ]],
            ['mask', ['recurrent_stack', ]],
            ['estimates', ['mask', 'audio']],
            ['audio', ['estimates', {'direction': 'inverse'}]]
        ]

    # define the outputs
    output = ['audio', 'mask']

    # put it together
    config = {
        'name': 'EndToEndRNN',
        'modules': modules,
        'connections': connections,
        'output': output
    }

    return config


def build_dual_path_recurrent_end_to_end(
        num_filters, filter_length, hop_length, 
        chunk_size, hop_size, hidden_size, num_layers, 
        bidirectional, bottleneck_size,
        num_sources, mask_activation, num_audio_channels=1,
        window_type='rectangular', skip_connection=False,
        embedding_bias=True, rnn_type='lstm', mix_key='mix_audio',
        init_forget=False):
    """
    Builds a config for a dual path recurrent network that operates on the 
    time-series. Uses a learned filterbank within the network.
    
    Args:
        num_filters (int): Number of learnable filters in the front end network.
        filter_length (int): Length of the filters.
        hop_length (int): Hop length between frames.
        window_type (str): Type of windowing function on apply to each frame.
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
        dict: A TASNet configuration that can be passed to
          SeparationModel.
    """
    
    # define the building blocks
    recurrent_hidden_size = hidden_size // 2 if bidirectional else hidden_size
    recurrent_stack = {
        'class': 'RecurrentStack',
        'args': {
            'num_features': bottleneck_size,
            'hidden_size': recurrent_hidden_size,
            'num_layers': 1,
            'bidirectional': bidirectional,
            'dropout': 0.0,
            'rnn_type': rnn_type,
            'batch_first': True,
            'init_forget': init_forget,
        }
    }

    modules = {
        mix_key: {},
        'audio': {
            'class': 'LearnedFilterBank',
            'args': {
                'num_filters': num_filters,
                'filter_length': filter_length,
                'hop_length': hop_length,
                'window_type': window_type,
                'requires_grad': True
            }
        },
        'mixture_weights': {
            'class': 'ReLU'
        },
        'dual_path': {
            'class': 'DualPath',
            'args': {
                'num_layers': num_layers,
                'chunk_size': chunk_size,
                'hop_size': hop_size,
                'skip_connection': skip_connection,
                'in_features': num_filters,
                'bottleneck_size': bottleneck_size,
                # rest are args to DualPathBlock
                'hidden_size': hidden_size,
                'intra_processor': recurrent_stack,
                'inter_processor': recurrent_stack,
            } 
        },
        'mask': {
            'class': 'Embedding',
            'args': {
                'num_features': num_filters,
                'hidden_size': num_filters,
                'embedding_size': num_sources,
                'activation': mask_activation,
                'num_audio_channels': num_audio_channels,
                'bias': embedding_bias,
                'dim_to_embed': [2, 3],
            }
        },
        'estimates': {
            'class': 'Mask',
        },
    }
    
    # define the topology
    connections = [
        ['audio', [mix_key, {'direction': 'transform'}]],
        ['mixture_weights', ['audio']],
        ['dual_path', ['mixture_weights', ]],
        ['mask', ['dual_path', ]],
        ['estimates', ['mask', 'mixture_weights']],
        ['audio', ['estimates', {'direction': 'inverse'}]]
    ]

    # define the outputs
    output = ['audio', 'mask']

    # put it together
    config = {
        'name': 'DualPathRNN',
        'modules': modules,
        'connections': connections,
        'output': output
    }

    return config
