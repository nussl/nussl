import torch

from ..base import ClusteringSeparationBase, DeepMixin, SeparationException


class DeepClustering(DeepMixin, ClusteringSeparationBase):
    """
    Clusters the embedding produced by a deep model for every time-frequency point.
    This is the deep clustering source separation approach. It is flexible with
    the number of sources. It expects that the model outputs a dictionary where one
    of the keys is 'embedding'. This uses the `DeepMixin` class to load the model
    and set the audio signal's parameters to be appropriate for the model.
    
    Args:
        input_audio_signal: (AudioSignal`) An AudioSignal object containing the 
          mixture to be separated.
        num_sources (int): Number of sources to cluster the features of and separate
          the mixture.
        model_path (str, optional): Path to the model that will be used. Can be None, 
          so that you can initialize a class and load the model later.  
          Defaults to None.
        device (str, optional): Device to put the model on. Defaults to 'cpu'.
        extra_data (dict, optional): Any extra data that is to be passed at runtime
          to the SeparationModel.
        **kwargs (dict): Keyword arguments for ClusteringSeparationBase and the 
          clustering object used for clustering (one of KMeans, GaussianMixture,
          MiniBatchKmeans).
    
    Raises:
        SeparationException: If 'embedding' isn't in the output of the model.
    """
    def __init__(self, input_audio_signal, num_sources, model_path=None,
                 device='cpu', extra_data=None, **kwargs):
        super().__init__(input_audio_signal, num_sources, **kwargs)
        if model_path is not None:
            self.load_model(model_path, device=device)
        # audio channel dimension in a dpcl model
        self.channel_dim = -1
        self.extra_data = extra_data

    def forward(self):
        return self.extract_features()

    def extract_features(self):
        input_data = self._get_input_data_for_model(self.extra_data)
        with torch.no_grad():
            output = self.model(input_data)
            if 'embedding' not in output:
                raise SeparationException(
                    "This model is not a deep clustering model! "
                    "Did not find 'embedding' key in output dictionary.")
            embedding = output['embedding']
            # swap back batch and sample dims
            if self.metadata['num_channels'] == 1:
                embedding = embedding.transpose(0, -2)
            embedding = embedding.squeeze(0).transpose(0, 1)
        self._preprocess_audio_signal()
        return embedding.cpu().data.numpy()
