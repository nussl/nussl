from ...ml import SeparationModel
from ...datasets import transforms as tfm
import torch

OMITTED_TRANSFORMS = (
    tfm.GetExcerpt, 
    tfm.MagnitudeWeights, 
    tfm.SumSources,
    tfm.Cache
)

class DeepMixin():
    def load_model(self, model_path, device='cpu'):
        """
        Loads the model at specified path `model_path`. Uses GPU if
        available.

        Args:
            model_path (str): path to model saved as SeparatonModel.
            device (str or torch.Device): loads model on CPU or GPU. Defaults to
              'cuda'.
              
        Returns:
            model (SeparationModel): Loaded model, nn.Module
            metadata (dict): metadata associated with model, used for making
            the input data into the model.
        """
        model_dict = torch.load(model_path, map_location='cpu')
        model =  SeparationModel(model_dict['config'])
        model.load_state_dict(model_dict['state_dict'])

        if not torch.cuda.is_available():
            device = 'cpu'

        self.device = device
        
        model = model.to(device).eval()
        metadata = model_dict['metadata'] if 'metadata' in model_dict else {}
        self.model = model
        self.metadata = metadata
        self.transform = self._get_transforms(
            self.metadata['transforms'])

    def _get_transforms(self, loaded_tfm):
        """
        Look through the loaded transforms and omits any that are in 
        `OMITTED_TRANSFORMS`.
        
        Args:
            loaded_tfm (Transform): A Transform from `nussl.datasets.transforms`.
        
        Returns:
            Transform: If the transform was a Compose, this returns a new Compose that
              omits the transforms listed in `OMITTED_TRANSFORMS`.
        """
        if isinstance(loaded_tfm, tfm.Compose):
            transform = []
            for _tfm in loaded_tfm.transforms:
                if not isinstance(_tfm, OMITTED_TRANSFORMS):
                    transform.append(_tfm)
            transform = tfm.Compose(transform)
        else:
            if not isinstance(loaded_tfm, OMITTED_TRANSFORMS):
                transform = loaded_tfm
            else:
                transform = None
        return transform
    
    def _get_input_data_for_model(self):
        """
        Sets up the audio signal with the appropriate STFT parameters and runs it
        through the transform found in the metadata.
        
        Returns:
            dict: Data dictionary to pass into the model.
        """
        if self.metadata['sample_rate'] is not None:
            if self.audio_signal.sample_rate != self.metadata['sample_rate']:
                self.audio_signal.resample(self.metadata['sample_rate'])
        
        self.audio_signal.stft_params = self.metadata['stft_params']

        data = {'mix': self.audio_signal}
        data = self.transform(data)

        for key in data:
            if torch.is_tensor(data[key]):
                data[key] = data[key].unsqueeze(0).to(self.device)
                if self.metadata['num_channels'] == 1:
                    # then each channel is processed indep
                    data[key] = data[key].transpose(0, -1)
        self.input_data = data
        return self.input_data
        