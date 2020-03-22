from ..base import MaskSeparationBase, DeepMixin, SeparationException
import numpy as np
import torch
from copy import deepcopy

class DeepMaskEstimation(MaskSeparationBase, DeepMixin):
    """
    Separates an audio signal using the masks produced by a deep model for every 
    time-frequency point. It expects that the model outputs a dictionary where one
    of the keys is 'masks'. This uses the `DeepMixin` class to load the model
    and set the audio signal's parameters to be appropriate for the model.
    
    Args:
        input_audio_signal: (AudioSignal`) An AudioSignal object containing the 
          mixture to be separated.        
        model_path (str, optional): Path to the model that will be used. Can be None, 
          so that you can initialize a class and load the model later.  
          Defaults to None.
        device (str, optional): Device to put the model on. Defaults to 'cpu'.
        **kwargs (dict): Keyword arguments for MaskSeparationBase.
    """
    def __init__(self, input_audio_signal, model_path=None, device='cpu', **kwargs):
        if model_path is not None:
            self.load_model(model_path, device=device)
        super().__init__(input_audio_signal, **kwargs)

    def forward(self):
        input_data = self._get_input_data_for_model()
        with torch.no_grad():
            output = self.model(input_data)
            if 'mask' not in output:
                raise SeparationException(
                    "This model is not a deep mask estimation model! "
                    "Did not find 'mask' key in output dictionary.")
            masks = output['mask']
            # swap back batch and sample dims
            if self.metadata['num_channels'] == 1:
                masks = masks.transpose(0, -2)
            masks = masks.squeeze(0).transpose(0, 1)
            masks = masks.cpu().data.numpy()
        self.model_output = output
        return masks

    def run(self, masks=None):
        self.result_masks = []

        if masks is None:
            masks = self.forward()

        for i in range(masks.shape[-1]):
            mask_data = masks[..., i]
            if self.mask_type == self.MASKS['binary']:
                mask_data = masks[..., i] == masks.max(axis=-1)
            mask = self.mask_type(mask_data)
            self.result_masks.append(mask)
        
        return self.result_masks
