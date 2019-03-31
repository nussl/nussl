def build_dpcl_conv(options):
    options['num_frequencies'] += 1
    options['num_features'] = (
        options['num_mels']
        if options['num_mels'] > 0
        else options['num_frequencies']
    )

    return {
        'modules': {
            'log_spectrogram': {
                'input_shape': (-1, -1, options['num_frequencies'])
            },
            'mel_projection': {
                'class': 'MelProjection',
                'args': {
                    'sample_rate': options['sample_rate'],
                    'num_frequencies': options['num_frequencies'],
                    'num_mels': options['num_mels'],
                    'direction': 'forward',
                    'trainable': options['projection_trainable'],
                    'clamp': False
                }
            },
            'dilated_convolutional_stack': {
                'class': 'DilatedConvolutionalStack',
                'args': {
                    'in_channels': options['in_channels'],
                    'channels': options['channels'],
                    'residuals': options['residuals'],
                    'dilations': options['dilations'],
                    'filter_shapes': options['filter_shapes'],
                    'batch_norm': options['batch_norm'],
                }
            },
            'embedding': {
                'class': 'Embedding',
                'args': {
                    'num_features': 1,
                    'hidden_size': options['channels'][-1],
                    'embedding_size': options['embedding_size'],
                    'activation': options['embedding_activations']
                }
            },
        },
        'connections': [
            ('mel_projection', ['log_spectrogram']),
            ('dilated_convolutional_stack', ['mel_projection']),
            ('embedding', ['dilated_convolutional_stack'])
        ],
        'output': ['embedding']
    }
