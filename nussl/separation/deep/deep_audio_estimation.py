import torch
from ..base import SeparationBase, DeepMixin, SeparationException


class DeepAudioEstimation(DeepMixin, SeparationBase):
    """
    Separates an audio signal using a model that produces separated sources directly
    in the waveform domain. It expects that the model outputs a dictionary where one
    of the keys is 'audio'. This uses the `DeepMixin` class to load the model
    and set the audio signal's parameters to be appropriate for the model.
    
    Args:
        input_audio_signal: (AudioSignal`) An AudioSignal object containing the 
          mixture to be separated.        
        model_path (str, optional): Path to the model that will be used. Can be None, 
          so that you can initialize a class and load the model later.  
          Defaults to None.
        extra_data: A dictionary containing any additional data that will 
          be merged with the output dictionary.
        device (str, optional): Device to put the model on. Defaults to 'cpu'.
        **kwargs (dict): Keyword arguments for MaskSeparationBase.
    """
    def __init__(self, input_audio_signal, model_path=None, device='cpu', 
                 extra_data=None, **kwargs):
        super().__init__(input_audio_signal, **kwargs)
        if model_path is not None:
            self.load_model(model_path, device=device)
        self.model_output = None
        self.extra_data = extra_data
        # audio channel dimension in an audio model
        self.channel_dim = 1

    def forward(self):
        input_data = self._get_input_data_for_model(self.extra_data)
        with torch.no_grad():
            output = self.model(input_data)
            if 'audio' not in output:
                raise SeparationException(
                    "This model is not a deep audio estimation model! "
                    "Did not find 'audio' key in output dictionary.")
            audio = output['audio']
            # swap back batch and sample dims
            if self.metadata['num_channels'] == 1:
                audio = audio.transpose(0, self.channel_dim)
            audio = audio.squeeze(0)
            audio = audio.cpu().data.numpy()
        self.model_output = output
        return audio

    def run(self, audio=None):
        if audio is None:
            audio = self.forward()
        self.audio = audio
        return self.audio

    def make_audio_signals(self):
        estimates = []
        for i in range(self.audio.shape[-1]):
            _estimate = self.audio_signal.make_copy_with_audio_data(
                self.audio[..., i])
            estimates.append(_estimate)
        return estimates
