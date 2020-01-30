import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from .. import networks
from .enums import *
import os
import shutil
from typing import Optional
from itertools import chain
import sys
import time
import time
import logging

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    SummaryWriter = None

class Trainer():
    def __init__( 
        self,
        output_folder,
        train_data,
        model,
        options,
        validation_data=None,
        use_tensorboard=True,
        experiment=None,
    ):
        self.options = options.copy()
        self.use_tensorboard = (
            use_tensorboard if SummaryWriter is not None else False
        )
        self.experiment = experiment
        self.prepare_directories(output_folder)
        self.log_frequency = self.options.pop('log_frequency', None)

        self.model = self.build_model(model)

        freeze_layers = self.options.pop('freeze_layers', [])
        self.freeze_layers(freeze_layers)
        
        self.device = torch.device(
            'cpu'
            if options['device'] == 'cpu' or not torch.cuda.is_available()
            else 'cuda'
        )
        self.model = self.model.to(self.device)

        if SummaryWriter is not None and use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.output_folder)
        else:
            self.writer = None
                    
        output_target_map = {
            'estimates': ['source_spectrograms'],
            'embedding': ['assignments', 'weights']
        }
        self.setup_loss(LossFunctions, output_target_map)
        self.model_tag = self.options.pop('model_tag', None)
        self.num_epoch = 0

        self.optimizer, self.scheduler = self.create_optimizer_and_scheduler(
            self.model,
            self.options
        )

        self.module = self.model
        resume = self.options.pop('resume', False)
        resume_location = self.options.pop('resume_location', None)
        resume_prefix = self.options.pop('resume_prefix', ('latest',))
        load_only_model = self.options.pop('load_only_model', False)
        if resume:
            self.resume(resume_location, load_only_model=load_only_model, prefixes=resume_prefix)

        if options['data_parallel'] and options['device'] == 'cuda':
            self.model = nn.DataParallel(self.module)
            self.module = self.model.module
        self.model.train()

        self.dataloaders = {
            'training': self.create_dataloader(train_data),
        }
        if validation_data:
            self.dataloaders['validation'] = self.create_dataloader(
                validation_data
            )

    def setup_loss(self, loss_functions, output_target_map):
        self.output_target_map = output_target_map
        self.loss_dictionary = {}
        for (_fn, target, weights) in self.options['loss_function']:
            if 'PIT' in _fn.upper() or 'CIT' in _fn.upper():
                it_loss, loss_fn = _fn.split(':')
                loss_fn = loss_functions[loss_fn.upper()].value()
                fn = loss_functions[it_loss.upper()].value(loss_fn)
            else:
                fn = loss_functions[_fn.upper()].value()
            self.loss_dictionary[target] = (fn, float(weights))
        self.loss_keys = sorted(list(self.loss_dictionary))

    @staticmethod
    def build_model(model):
        return (
            networks.SeparationModel(model)
            if (type(model) is str and '.json' in model) or type(model) is dict
            else model
        )

    def freeze_layers(self, freeze_layers):
        for name, parameter in self.model.named_parameters():
            if name in freeze_layers:
                logging.info(f'Freezing {name} layer in model.')
                parameter.requires_grad = False

    def prepare_directories(self, output_folder):
        self.output_folder = output_folder
        self.checkpoint_folder = os.path.join(output_folder, 'checkpoints')

        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.checkpoint_folder, exist_ok=True)

    def create_dataloader(self, dataset):
        if not dataset:
            return None

        # input_keys = []
        # for connection in self.module.connections:
        #     for c in connection:
        #         if isinstance(c, dict):
        #             for key, val in c.items():
        #                 input_keys.append(val)
        #         else:
        #             input_keys.append(c)

        # input_keys += self.module.output_keys
    

        # output_keys = []
        # for k in input_keys:
        #     print(k)
        #     if k in self.output_target_map:
        #         output_keys.append(self.output_target_map[k])

        # dataset.data_keys_for_training = input_keys + output_keys

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

    def preprocess_targets(self, targets):
        if self.module.layers['mel_projection'].num_mels > 0:
            if 'assignments' in targets:
                targets['assignments'] = self.module.project_data(
                    targets['assignments'],
                    clamp=True
                )
                targets['assignments'] = (targets['assignments'] > 0).float()
            if 'weights' in targets:
                targets['weights'] = self.module.project_data(
                    targets['weights'],
                    clamp=False
                )
                if 'threshold' in self.dataloaders['training'].dataset.options['weight_type']:
                    targets['weights'] = (targets['weights'] > 0).float()
        return targets

    def calculate_loss(self, outputs, targets):
        targets = self.preprocess_targets(targets)
        loss = {}
        for key in self.loss_keys:
            loss_function = self.loss_dictionary[key][0]
            weight = self.loss_dictionary[key][1]
            target_keys = self.output_target_map[key]
            arguments = [outputs[key]] + [targets[t] for t in target_keys]
            _loss = weight * loss_function(*arguments)
            self.check_loss(_loss)
            loss[key] = _loss
        return loss

    def check_loss(self, loss):
        if np.isnan(loss.item()):
            raise ValueError("Loss went to nan - aborting training.")

    def prepare_data(self, data):
        """Moves all the data to the requested device (cuda or cpu).
        """
        for key in data:
            data[key] = data[key].float().to(self.device)
            if key not in self.loss_keys and self.model.training:
                data[key] = data[key].requires_grad_()
        return data
    
    def forward(self, data):
        """Performs a forward pass on the data using the model.
        """
        if self.model.training:
            output = self.model(data)
        else:
            output = self.module(data)
        return output

    def run_epoch(self, key):
        epoch_loss = 0
        num_batches = len(self.dataloaders[key])
        for step, data in enumerate(self.dataloaders[key]):
            #TODO factor this out into a loop that can be overridden by other classes.
            data = self.prepare_data(data)
            output = self.forward(data)
            loss = self.calculate_loss(output, data)
            loss['total_loss'] = sum(list(loss.values()))
            epoch_loss += loss['total_loss'].item()

            if self.log_frequency and step % self.log_frequency == 0 and step > 0:
                logging.info(f"At {step}/{num_batches} with {loss['total_loss'].item()}")

            if self.model.training:
                self.optimizer.zero_grad()
                loss['total_loss'].backward()
                self.optimizer.step()
            step += 1
        return {'loss': epoch_loss / float(num_batches)}

    def before_epoch(self):
        """
        Do whatever needs to be done before each epoch (e.g. lengthen sequences).
        Can use this to implement curriculum learning.
        """
        if self.options['curriculum_learning']:
            settings = self.options['curriculum_learning']
            for setting in settings:
                if setting['num_epoch'] == self.num_epoch:
                    logging.info(
                        f"Curriculum learning - running @ epoch: {setting['num_epoch']}, "
                        f"command: {setting['command']}, "
                        f"args: {setting['args']}"
                    )
                    if setting['command'] == 'set_current_length':
                        for key, dataloader in self.dataloaders.items():
                            dataloader.dataset.set_current_length(*setting['args'])
                    if setting['command'] == 'save':
                        self.save(is_best=False, prefix=f'epoch{self.num_epoch}')

    def log_metrics(self, data, step, scope):
        if self.use_tensorboard:
            prefix = 'training' if self.model.training else 'validation'
            for key in data:
                label = os.path.join(prefix, key)
                self.writer.add_scalar(label, data[key], step)
        if self.experiment is not None:
            context = (
                self.experiment.train 
                if self.model.training 
                else self.experiment.validate
            )
            with context():
                for key in data:
                    self.experiment.log_metric(key, data[key], step)

    def populate_cache(self):
        for key, dataloader in self.dataloaders.items():
            num_batches = len(dataloader)
            if dataloader.dataset.overwrite_cache:
                logging.info(f'Populating cache for {key} w/ {num_batches} batches')
                for i, _ in enumerate(dataloader):
                    if i % 50 == 0:
                        logging.info(f'{i}/{num_batches} batches completed')
                    continue
                logging.info(f'Done populating cache for {key}')

    def switch_to_cache(self):
        for key, dataloader in self.dataloaders.items():
            dataloader.dataset.switch_to_cache()
            
    def fit(self):
        lowest_validation_loss = np.inf

        self.populate_cache()
        self.switch_to_cache()

        logging.info(f"Training for {self.options['num_epochs'] - self.num_epoch} epochs")
        fit_start_time = time.time()

        for self.num_epoch in range(self.num_epoch, self.options['num_epochs']):
            self.before_epoch()
            start_time = time.time()
            epoch_loss = self.run_epoch('training')
            self.log_metrics(epoch_loss, self.num_epoch, 'epoch')
            validation_loss = self.validate('validation')
            saved_model_path = self.save(validation_loss['loss'] < lowest_validation_loss)
            lowest_validation_loss = (
                validation_loss['loss'] 
                if validation_loss['loss'] < lowest_validation_loss
                else lowest_validation_loss
            )

            epoch_elapsed_time = time.time() - start_time
            epoch_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(epoch_elapsed_time))
            full_elapsed_time = time.time() - fit_start_time
            full_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(full_elapsed_time))

            logging_str = (
                f"""\n
                EPOCH SUMMARY
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                | Epoch number: {self.num_epoch:04d}         |                
                | Training loss:   {epoch_loss['loss']:04f}  |           
                | Validation loss: {validation_loss['loss']:04f}  |   
                | Epoch took: {epoch_elapsed_time}       |                 
                | Time since start: {full_elapsed_time} |          
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                Saving to {saved_model_path}.
                Configuration @ {self.output_folder}
                """
            )

            if hasattr(self.experiment, '_get_experiment_url'):
                # then this is a comet.ml run with a link
                logging_str += f'Watch experiment @ {self.experiment._get_experiment_url()}\n'

            logging.info(logging_str)

    def validate(self, key) -> float:
        """Calculate loss on validation set

        Args:
            dataloader - a dataloader yielding the validation data if there is
                any, `None` otherwise

        Returns:
            `np.inf` if there is no validation dataset (`dataloader` is `None`)
            else the loss over the given validation data
        """
        if key not in self.dataloaders:
            return {'loss': np.inf}

        self.model.eval()
        with torch.no_grad():
            validation_loss = self.run_epoch(key)
        self.log_metrics(validation_loss, self.num_epoch, 'epoch')
        self.model.train()
        self.scheduler.step(validation_loss['loss'])
        if self.scheduler.num_bad_epochs + 1 == self.options['patience']:
            logging.info(f'Exceeded patience at epoch {self.num_epoch} - adjusting learning rate')   
            
        return validation_loss

    def save(self, is_best: bool, prefix: str = '') -> str:
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
        if not prefix:
            prefix = 'latest'
        if self.model_tag is not None:
            prefix = f'{prefix}.{self.model_tag}'
        optimizer_path = os.path.join(self.checkpoint_folder, f'{prefix}.opt.pth')
        model_path = os.path.join(
             self.checkpoint_folder,
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
        if is_best:
            torch.save(optimizer_state, optimizer_path.replace('latest', 'best'))
            model_path = model_path.replace('latest', 'best')
            self.module.save(model_path, {'metadata': metadata})
        return model_path

    def resume(self, resume_location, load_only_model=True, prefixes=('best', 'latest')):
        for prefix in prefixes:
            checkpoint_folder = self.checkpoint_folder
            if resume_location is not None:
                checkpoint_folder = resume_location
            
            optimizer_path = os.path.join(checkpoint_folder, f'{prefix}.opt.pth')
            model_path = os.path.join(checkpoint_folder, f'{prefix}.model.pth')
            logging.info(f'Looking for {model_path}')

            if os.path.exists(model_path) and os.path.exists(optimizer_path):
                optimizer_state = torch.load(optimizer_path, map_location=lambda storage, loc: storage)
                model_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
                self.module.load_state_dict(model_dict['state_dict'])
                if not load_only_model:
                    self.optimizer.load_state_dict(optimizer_state['optimizer'])
                    self.num_epoch = optimizer_state['num_epoch']
                logging.info(f'Resuming from {model_path}')
                return model_path
            else:
                model_path = None
        logging.info('No model found! Training a new model from scratch.')
        return model_path