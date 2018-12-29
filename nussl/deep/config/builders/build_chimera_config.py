def build_chimera_config(options):
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
            'magnitude_spectrogram': {
                'input_shape': (-1, -1, options['num_frequencies'])
            },
            'mel_projection': {
                'class': 'MelProjection',
                'args': {
                    'sample_rate': options['sample_rate'],
                    'num_frequencies': options['num_frequencies'],
                    'num_mels': options['num_mels'],
                    'direction': 'forward',
                    'trainable': options['trainable'],
                    'clamp': False
                }
            },
            'recurrent_stack': {
                'class': 'RecurrentStack',
                'args': {
                    'num_features': options['num_features'],
                    'hidden_size': options['hidden_size'],
                    'num_layers': options['num_layers'],
                    'bidirectional': options['bidirectional'],
                    'dropout': options['dropout'],
                    'rnn_type': options['rnn_type']
                }
            },
            'embedding': {
                'class': 'Embedding',
                'args': {
                    'num_features': options['num_features'],
                    'hidden_size': (
                        2 * options['hidden_size']
                        if options['bidirectional']
                        else options['hidden_size']
                    ),
                    'embedding_size': options['embedding_size'],
                    'activation': options['embedding_activations']
                }
            },
            'masks': {
                'class': 'Embedding',
                'args': {
                    'num_features': options['num_features'],
                    'hidden_size': (
                        2 * options['hidden_size']
                        if options['bidirectional']
                        else options['hidden_size']
                    ),
                    'embedding_size': options['num_sources'],
                    'activation': options['mask_activation']
                }
            },
            'inv_projection': {
                'class': 'MelProjection',
                'args': {
                    'sample_rate': options['sample_rate'],
                    'num_frequencies': options['num_frequencies'],
                    'num_mels': options['num_mels'],
                    'direction': 'backward',
                    'clamp': True
                }
            },
            'estimates': {
                'class': 'Mask',
                'args': {}
            }
        },
        'connections': [
            ('mel_projection', ['log_spectrogram']),
            ('recurrent_stack', ['mel_projection']),
            ('embedding', ['recurrent_stack']),
            ('masks', ['recurrent_stack']),
            ('inv_projection', ['masks']),
            ('estimates', ['inv_projection', 'magnitude_spectrogram'])
        ],
        'output': ['embedding', 'estimates']
    }
