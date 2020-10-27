import torch
import yaml
import json

from ...ml import SeparationModel
from ...datasets import transforms as tfm
from ...core.migration import SafeModelLoader

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
        safe_loader = SafeModelLoader()
        model_dict = safe_loader.load(model_path, 'cpu')
        metadata = model_dict['metadata']

        model = SeparationModel(metadata['config'])
        model.load_state_dict(model_dict['state_dict'])
        device = device if torch.cuda.is_available() else 'cpu'

        self.device = device

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

    def _get_input_data_for_model(self, extra_data=None):
        """
        Sets up the audio signal with the appropriate STFT parameters and runs it
        through the transform found in the metadata.

        Args:
            extra_data: A dictionary containing any additional data that will 
              be merged with the output dictionary.
        
        Returns:
            dict: Data dictionary to pass into the model.
        """
        extra_data = {} if extra_data is None else extra_data
        if self.metadata['sample_rate'] is not None:
            if self.audio_signal.sample_rate != self.metadata['sample_rate']:
                self.audio_signal.resample(self.metadata['sample_rate'])

        self.audio_signal.stft_params = self.metadata['stft_params']
        self.audio_signal.stft()

        data = {'mix': self.audio_signal}
        data.update(extra_data)
        data = self.transform(data)

        for key in data:
            if torch.is_tensor(data[key]):
                data[key] = data[key].unsqueeze(0).to(self.device).float()
                if self.metadata['num_channels'] == 1:
                    # then each channel is processed indep
                    data[key] = data[key].transpose(0, self.channel_dim)
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
