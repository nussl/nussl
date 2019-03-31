import torch
import numpy as np
import librosa
from ..deep import SeparationModel

class DeepMixin():
    def load_model(self, model_path):
        """
        Loads the model at specified path `model_path`

        Args:
            model_path (str): path to model saved as SeparatonModel

        Returns:
            model (SeparationModel): Loaded model, nn.Module
            metadata (dict): metadata associated with model, used for making
            the input data into the model.

        """
        model_dict = torch.load(model_path, map_location='cpu')
        model = SeparationModel(model_dict['config'])
        model.load_state_dict(model_dict['state_dict'])
        model = model.to(self.device).eval()
        metadata = model_dict['metadata'] if 'metadata' in model_dict else {}
        return model, metadata

    def _preprocess(self):
        """
        Preprocess data before putting into the model.
        """
        data = {
            'magnitude_spectrogram': np.abs(self.stft),
            'log_spectrogram': self.log_spectrogram.copy(),
        }
        data['log_spectrogram'] -= np.mean(data['log_spectrogram'])
        data['log_spectrogram'] /= np.std(data['log_spectrogram']) + 1e-7

        for key in data:
            data[key] = self._format(data[key], self.metadata['format'])
            data[key] = torch.from_numpy(data[key]).to(self.device).float()

        return data

    def _format(
        self,
        datum: np.ndarray,
        format_type: str,
    ) -> np.ndarray:
        """
        Formats data for input to model

        Formats produced based on `format_type`:
            - `rnn`
                [num_batch, sequence_length, num_frequencies*num_channels, ...]
                [num_batch, num_channels, sequence_length, num_frequencies, ...]
            - `cnn`
                [num_batch, num_channels, num_frequencies, sequence_length, ...]

        Args:
            datum - numpy array holding input data to be formatted
            format_type - indicates the architecture (and therefore necessary
                formatting). Choices: ['rnn', 'cnn'].
                TODO: make this an enum, not a `str`

        Returns:
            formatted datum
        """
        if format_type == 'rnn':
            datum = np.expand_dims(datum, axis=0)
            if self.metadata['num_channels'] != self.audio_signal.num_channels:
                datum = np.swapaxes(datum, 0, 3)

            _shape = datum.shape
            shape = [_shape[0], _shape[1], _shape[2], _shape[3]]
            datum = np.reshape(datum, shape)
            datum = np.swapaxes(datum, 1, 2)
        elif format_type == 'cnn':
            datum = np.moveaxis(datum, [0, 1, 2, 3], [0, 3, 2, 1])
        else:
            raise Exception(f'Unexpected format type: {format_type}')

        return datum