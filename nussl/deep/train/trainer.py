import torch
from torch import nn
from tqdm import trange, tqdm
import multiprocessing
from torch.utils.data import DataLoader
import numpy as np
from .. import networks
from .enums import *
import os
import shutil
from typing import Optional
from itertools import chain
tqdm.monitor_interval = 0

try:
    from tensorboardX import SummaryWriter
except:
    SummaryWriter = None


OutputTargetMap = {
    'estimates': ['source_spectrograms'],
    'embedding': ['assignments', 'weights']
}

class Trainer():
    def __init__( 
        self,
        output_folder,
        train_data,
        model,
        options,
        validation_data=None,
        use_tensorboard=True,
    ):
        self.use_tensorboard = use_tensorboard
        self.prepare_directories(output_folder)
        self.model = self.build_model(model)
        self.device = torch.device(
            'cpu'
            if options['device'] == 'cpu' or not torch.cuda.is_available()
            else 'cuda'
        )
        self.model = self.model.to(self.device)
        print(self.model)

        self.writer = (
            SummaryWriter(log_dir=self.output_folder)
            if use_tensorboard and SummaryWriter is not None
            else None
        )
        self.loss_dictionary = {
            target: (LossFunctions[fn.upper()].value(), float(weight))
            for (fn, target, weight)
            in options['loss_function']
        }
        self.loss_keys = sorted(list(self.loss_dictionary))
        self.options = options
        self.num_epoch = 0

        self.optimizer, self.scheduler = self.create_optimizer_and_scheduler(
            self.model,
            self.options
        )
        self.module = self.model
        if options['data_parallel'] and options['device'] == 'cuda':
            self.model = nn.DataParallel(self.model)
            self.module = self.model.module
        self.model.train()

        self.dataloaders = {
            'training': self.create_dataloader(train_data),
        }
        if validation_data:
            self.dataloaders['validation'] = self.create_dataloader(
                validation_data
            )

    @staticmethod
    def build_model(model):
        return (
            networks.SeparationModel(model)
            if (type(model) is str and '.json' in model) or type(model) is dict
            else model
        )

    def prepare_directories(self, output_folder):
        self.output_folder = output_folder
        self.checkpoint_folder = os.path.join(output_folder, 'checkpoints')
        self.config_folder = os.path.join(output_folder, 'config')

        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        os.makedirs(self.config_folder, exist_ok=True)

    def create_dataloader(self, dataset):
        if not dataset:
            return None

        input_keys = [[connection[0]] + connection[1] for connection in self.module.connections]
        input_keys = list(chain.from_iterable(input_keys))
        input_keys += self.module.output_keys

        output_keys = [OutputTargetMap[k] for k in input_keys if k in OutputTargetMap]
        output_keys = list(chain.from_iterable(output_keys))
        
        dataset.data_keys_for_training = input_keys + output_keys

        return DataLoader(
            dataset,
            batch_size=self.options['batch_size'],
            num_workers=self.options['num_workers'],
            sampler=Samplers[
                self.options['sample_strategy'].upper()
            ].value(dataset),
            pin_memory=True,
        )

    def create_optimizer_and_scheduler(self, model, options):
        parameters = filter(lambda p: p.requires_grad, model.parameters())

        optimizer = Optimizers[options['optimizer'].upper()].value(
            parameters,
            lr=options['learning_rate'],
            weight_decay=options['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            factor=options['learning_rate_decay'],
            patience=options['patience'],
        )
        return optimizer, scheduler

    def calculate_loss(self, outputs, targets):
        if self.module.layers['mel_projection'].num_mels > 0:
            if 'assignments' in targets:
                targets['assignments'] = self.module.project_data(
                    targets['assignments'],
                    clamp=True
                )
            if 'weights' in targets:
                targets['weights'] = self.module.project_data(
                    targets['weights'],
                    clamp=False
                )
        loss = {}
        for key in self.loss_keys:
            loss_function = self.loss_dictionary[key][0]
            weight = self.loss_dictionary[key][1]
            target_keys = OutputTargetMap[key]
            arguments = [outputs[key]] + [targets[t] for t in target_keys]
            _loss = weight * loss_function(*arguments)
            self.check_loss(_loss)
            loss[key] = _loss
        return loss

    def check_loss(self, loss):
        if np.isnan(loss.item()):
            raise ValueError("Loss went to nan - aborting training.")

    def run_epoch(self, dataloader):
        epoch_loss = 0
        step = 0
        num_batches = len(dataloader)
        progress_bar = trange(0, num_batches)
        for data in dataloader:
            for key in data:
                data[key] = data[key].float().to(self.device)
                if key not in self.loss_keys and self.model.training:
                    data[key] = data[key].requires_grad_()
            output = self.model(data)
            loss = self.calculate_loss(output, data)
            loss['total_loss'] = sum(list(loss.values()))
            epoch_loss += loss['total_loss'].item()

            if self.model.training:
                self.log_to_tensorboard(
                    {k: loss[k].item() for k in loss},
                    step + self.num_epoch * num_batches,
                    'iter'
                )
                self.optimizer.zero_grad()
                loss['total_loss'].backward()
                self.optimizer.step()
            progress_bar.update(1)
            progress_bar.set_description(f"Loss: {loss['total_loss']:.4f}")
            step += 1
        return {'loss': epoch_loss / float(num_batches)}

    def log_to_tensorboard(self, data, step, scope):
        if self.use_tensorboard:
            prefix = 'training' if self.model.training else 'validation'
            for key in data:
                label = os.path.join(prefix, key)
                self.writer.add_scalar(label, data[key], step)

    def fit(self):
        progress_bar = trange(self.num_epoch, self.options['num_epochs'])
        lowest_validation_loss = np.inf
        for self.num_epoch in progress_bar:
            epoch_loss = self.run_epoch(self.dataloaders['training'])
            self.log_to_tensorboard(epoch_loss, self.num_epoch, 'epoch')
            validation_loss = self.validate(self.dataloaders.get('validation'))
            self.save(validation_loss < lowest_validation_loss)
            lowest_validation_loss = (
                validation_loss 
                if validation_loss < lowest_validation_loss
                else lowest_validation_loss
            )

            progress_bar.update(1)
            progress_bar.set_description(f'Loss: {epoch_loss["loss"]:.4f}')

    def validate(self, dataloader: Optional[DataLoader]) -> float:
        """Calculate loss on validation set

        Args:
            dataloader - a dataloader yielding the validation data if there is
                any, `None` otherwise

        Returns:
            `np.inf` if there is no validation dataset (`dataloader` is `None`)
            else the loss over the given validation data
        """
        if not dataloader:
            return np.inf

        self.model.eval()
        with torch.no_grad():
            validation_loss = self.run_epoch(dataloader)
        self.log_to_tensorboard(validation_loss, self.num_epoch, 'epoch')
        self.model.train()
        self.scheduler.step(validation_loss['loss'])
        return validation_loss['loss']

    def save(self, is_best: bool, path: str = '') -> str:
        """Saves the model being trained with either `latest` or `best` prefix
        based on validation loss

        Args:
            is_best - whether or not the model is known to have improved
                accuracy from the last epoch (based on validation loss). Always
                False if no validation data was given.
            [path] - path to folder in which to save model. If not given, model
                saved in checkpoint folder

        Returns:
            path to saved model
        """
        if path:
            os.makedirs(path, exist_ok=True)

        prefix = 'best' if is_best else 'latest'
        optimizer_path = os.path.join(
            self.checkpoint_folder,
            f'{prefix}.opt.pth'
        )
        model_path = os.path.join(
             path if path else self.checkpoint_folder,
            f'{prefix}.model.pth'
        )
        dataset_options = self.dataloaders['training'].dataset.options
        metadata = {
            key: val
            for key, val in dataset_options.items()
            if key in [
                'n_fft',
                'hop_length',
                'format',
                'sample_rate',
                'num_channels'
            ]
        }

        optimizer_state = {
            'optimizer': self.optimizer.state_dict(),
            'num_epoch': self.num_epoch
        }

        torch.save(optimizer_state, optimizer_path)
        self.module.save(model_path, {'metadata': metadata})
        return model_path

    def resume(self, prefix='best'):
        optimizer_path = os.path.join(self.checkpoint_folder, f'{prefix}.opt.pth')
        model_path = os.path.join(self.checkpoint_folder, f'{prefix}.model.pth')

        optimizer_state = torch.load(optimizer_path)

        model_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        model = model_dict['model']
        model.load_state_dict(model_dict['state_dict'])

        self.model = self.load_model(model_path)
        return
