import torch
import yaml
import json

from ...ml import SeparationModel
from ...datasets import transforms as tfm

OMITTED_TRANSFORMS = (
    tfm.GetExcerpt,
    tfm.MagnitudeWeights,
    tfm.SumSources,
    tfm.Cache,
    tfm.IndexSources,
)


class DeepMixin:
    def load_model(self, model_path, device='cpu'):
        """
        Loads the model at specified path `model_path`. Uses GPU if
        available.

        Args:
            model_path (str): path to model saved as SeparationModel.
            device (str or torch.Device): loads model on CPU or GPU. Defaults to
              'cuda'.

        Returns:
            model (SeparationModel): Loaded model, nn.Module
            metadata (dict): metadata associated with model, used for making
            the input data into the model.
        """
        
        device = device if torch.cuda.is_available() else 'cpu'

        self.device = device
        model, metadata = SeparationModel.load(model_path)
        model = model.to(device).eval()
        self.model = model
        self.config = metadata['config']
        self.metadata.update(metadata)
        self.transform = self._get_transforms(metadata['train_dataset']['transforms'])

    @staticmethod
    def _get_transforms(loaded_tfm):
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

    def modify_input_data(self, data, **kwargs):
        """Add or modify input data to dictionary before passing 
        it to the model. By default this just updates the data
        dictionary with what is needed, but can be overridden
        by classes inheriting this method to modify the data
        dictionary as needed.

        Parameters
        ----------
        data : dict,
            The data dictionary before this function is called.
        kwargs : keyword arguments, optional
            Data dictionary after this function is called, by default None
        """
        data.update(kwargs)
        return data

    def _get_input_data_for_model(self, **kwargs):
        """
        Sets up the audio signal with the appropriate STFT parameters and runs it
        through the transform found in the metadata.

        Args:
            kwargs: Any additional data that will 
              be merged with the input dictionary.
        
        Returns:
            dict: Data dictionary to pass into the model.
        """
        if self.metadata['sample_rate'] is not None:
            if self.audio_signal.sample_rate != self.metadata['sample_rate']:
                self.audio_signal.resample(self.metadata['sample_rate'])

        self.audio_signal.stft_params = self.metadata['stft_params']
        self.audio_signal.stft()

        data = {'mix': self.audio_signal}
        data = self.transform(data)

        for key in data:
            if torch.is_tensor(data[key]):
                data[key] = data[key].unsqueeze(0).to(self.device).float()
                if self.metadata['num_channels'] == 1:
                    # then each channel is processed indep
                    data[key] = data[key].transpose(0, self.channel_dim)
        
        data = self.modify_input_data(data, **kwargs)       
        self.input_data = data
        return self.input_data

    def get_metadata(self, to_str=False, **kwargs):
        """
        Gets the metadata associated with this model.
        Args:
            to_str (bool): If True, will return a string, else will return dict.
            for_upload (bool): If True, will scrub metadata for uploading to EFZ.

        Returns:
            (str) or (dict) containing metadata.
        """
        for_upload = kwargs.get('for_upload', False)
        truncate_loss = kwargs.get('truncate_loss', False)
        metadata = getattr(self, 'metadata', None)

        if metadata is None:
            raise ValueError('Could not find associated metadata.')

        if for_upload:
            # remove paths
            keys = ['train_dataset', 'val_dataset']
            for k in keys:
                if k in metadata:
                    metadata[k].pop('folder')

        if for_upload or truncate_loss:
            if 'trainer.state.epoch_history' in metadata:
                loss_history = metadata.pop('trainer.state.epoch_history')
                metadata['final_loss'] = {k: float(v[-1]) for k, v in loss_history.items()}

        if isinstance(metadata['config'], str):
            metadata['config'] = json.loads(metadata['config'])
        metadata['separation_class'] = type(self).__name__
        metadata['model_name'] = metadata['config']['name']

        if to_str:
            return yaml.dump(metadata, indent=4)
        else:
            return metadata
