import ignite
from ignite.engine import Events, Engine
from ignite.handlers import Timer
import torch
import logging

def create_engine(output_folder, model, train_data, train_func, optimizer,
    validation_date=None, val_func=None, max_epochs=10, device='cpu', **kwargs):
    """
    Helper function for creating an ignite Engine object with helpful defaults.
    
    Args:
        output_folder ([type]): [description]
        model ([type]): [description]
        train_data ([type]): [description]
        train_func ([type]): [description]
        optimizer ([type]): [description]
        validation_date ([type], optional): [description]. Defaults to None.
        val_func ([type], optional): [description]. Defaults to None.
        max_epochs (int, optional): [description]. Defaults to 10.
        device (str, optional): [description]. Defaults to 'cpu'.
    """
    trainer = Engine(train_func)
    validator = Engine(val_func)

    overall_timer = Timer(average=False)
    overall_timer.attach(trainer, start=Events.STARTER, Events.COMPLETED)

    epoch_timer = Timer(average=False)
    epoch_timer.attach(trainer, start=Events.EPOCH_STARTED, end=Events.EPOCH_COMPLETED)

    @validator.on(Events.EPOCH_COMPLETED)
    def log_epoch_to_stdout(engine):
        epoch_time = epoch_timer.value()
        overall_time = overall_timer.value()
        epoch_number = trainer.state.epoch 
        validation_loss = validator.output.loss
        train_loss = trainer.output.loss
        saved_model_path = 

        logging_str = (
            f"""\n
            EPOCH SUMMARY
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            - Epoch number: {self.num_epoch:04d}                         
            - Training loss:   {train_loss['loss']:04f}         
            - Validation loss: {validation_loss['loss']:04f}     
            - Epoch took: {epoch_time}                
            - Time since start: {overall_time}       
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            Saving to {saved_model_path}.
            Output @ {output_folder}
            """
        )

        logging.info(logging_str)






        






class Trainer(object):
    def __init__(self, output_folder, model, train_data, train_func, optimizer, 
        validation_data=None, val_func=None, max_epochs=10, device='cpu', **kwargs):

        self._train_func = train_func
        self._val_func = val_func

        if device == 'cuda' and not torch.cuda.is_available():
            logging.warn(
                f"Device was set to {device} but CUDA not available. "
                f"Setting to 'cpu'."
            )

        self.device = torch.device(device)
        
        self.train_engine = Engine(self.train_func)
        self.val_engine = Engine(self.val_func)

    def prepare_data(self, data):
        for key in data:
            pass
