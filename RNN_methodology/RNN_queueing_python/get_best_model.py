import numpy as np
from keras.callbacks import Callback

"""This code has been inspired and adapted from the following source:
Garbi, G et al. (2020). Learning Queueing Networks by Recurrent Neural Networks.
Accessible at:
https://pdfs.semanticscholar.org/7f7c/12bcc23ba098ad5a4a0ad251bd92e9b9c27a.pdf."""

class GetBestModel(Callback):

    """For this script we create a class that can get the best model. We choose
    inputs that follow the same format for the 'EarlyStopping' method in Keras.
    As default, we monitor the validation loss. We are constrained on the methods
    that we use because we feed this function into a keras function when we fit
    our model to the data. The methods that we include are: 'set_weights_init',
    'on_epoch_end' and 'on_train_end'. We set this class such that we can only
    monitor the val loss and we keep the mode as 'min'. 'min' means we are trying
    to minimize the val loss. 'set_weights_init' takes the weights of the model
    that has given the minimum loss value; on_epoch_end shows the change in val_loss
    for each epoch; and on_train_end gives the best loss and the best epoch that
    it occurred."""

    def __init__(self, monitor='val_loss',verbose=0,mode='min',period=1):
        super(GetBestModel, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.best_epochs = 0
        self.last_save_epoch = 0

        if mode != 'min':
            print(f'mode {mode} is wrong for val_loss monitoring, changing it to min')

            mode = 'min'
        if self.monitor != 'val_loss':
            print(f'monitor value should be set to val_loss, currently it is {self.monitor}')
            self.monitor = 'val_loss'

        self.monitor_op = np.less
        self.best = np.Inf

    def set_weights_init(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.last_save_epoch += 1
        if self.last_save_epoch >= self.period:
            self.last_save_epoch = 0
            current = logs.get(self.monitor)
            if current is None:
                print(f'Can only pick best model when {self.monitor} is available')
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print(f'Epoch {epoch+1}: {self.monitor} improved from {self.best} to {current}, storing weights.')
                    self.best = current
                    self.best_epochs = epoch + 1
                    self.best_weights = self.model.get_weights()
                else:
                    if self.verbose > 0:
                        print(f'Epoch {epoch+1}: no improvement in{self.monitor}')

    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print(f'Using epoch {self.best_epochs} with {self.monitor}: {self.best}')

        self.model.set_weights(self.best_weights)
