import torch

from ..base import MaskSeparationBase, DeepMixin, SeparationException
from ... import ml


class DeepMaskEstimation(DeepMixin, MaskSeparationBase):
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
        extra_data: A dictionary containing any additional data that will 
          be merged with the output dictionary. This can come from a dataset,
          or contain a query, etc.
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
        # audio channel dimension in a mask estimation model
        self.channel_dim = -1

    def forward(self):
        input_data = self._get_input_data_for_model(self.extra_data)
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

    def confidence(self, approach='silhouette_confidence', num_sources=2, **kwargs):
        """
        In embedding-based separation algorithms, we can compute a confidence
        measure based on the clusterability of the embedding space. This can
        be used if the model also computes an embedding alongside the estimates
        (e.g. as in Chimera models.) 
        
        Args:
            approach (str, optional): What approach to use for getting the confidence 
              measure. Options are 'jensen_shannon_confidence', 'posterior_confidence', 
              'silhouette_confidence', 'loudness_confidence', 'whitened_kmeans_confidence',
              'dpcl_classic_confidence'. Defaults to 'silhouette_confidence'.
            kwargs: Keyword arguments to the function being used to compute the confidence.
        """
        if self.model_output is None:
            raise SeparationException(
                "self.model_output is None! Did you run forward?")
        if 'embedding' not in self.model_output:
            raise SeparationException(
                "embedding not in self.model_output! Can't compute confidence.")
        features = self.model_output['embedding']
        if self.metadata['num_channels'] == 1:
            features = features.transpose(0, -2)
        features = features.squeeze(0).transpose(0, 1)
        features = features.cpu().data.numpy()

        confidence_function = getattr(ml.confidence, approach)
        confidence = confidence_function(
            self.audio_signal, features, num_sources, **kwargs)
        return confidence
