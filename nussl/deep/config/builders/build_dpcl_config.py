def build_dpcl_config(options):
    options['num_frequencies'] += 1
    options['num_features'] = (
        options['num_mels']
        if options['num_mels'] > 0
        else options['num_frequencies']
    )
    options['num_features'] *= options['num_channels']
    
    norm_layer = 'batch_norm'
    if options['instance_norm']:
        norm_layer = 'instance_norm'

    return {
        'modules': {
            'log_spectrogram': {
                'input_shape': (-1, -1, options['num_frequencies'])
            },
            'batch_norm': {
                'class': 'BatchNorm',
                'args': {
                    'use_batch_norm': options['batch_norm']
                }
            },
            'instance_norm': {
                'class': 'InstanceNorm',
                'args': {
                    'use_instance_norm': options['instance_norm']
                }
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
                    'num_channels': options['num_channels'],
                    'hidden_size': (2 * options['hidden_size'] if options['bidirectional']
                                    else options['hidden_size']),
                    'embedding_size': options['embedding_size'],
                    'activation': options['embedding_activations']
                }
            },
        },
        'connections': [
            ('mel_projection', ['log_spectrogram']),
            (norm_layer, ['mel_projection']),
            ('recurrent_stack', [norm_layer]),
            ('embedding', ['recurrent_stack'])
        ],
        'output': ['embedding']
    }
