import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # I am assuming that x is an data sample and y is a tuple with the ground truth labels?
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called.
        # This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        if self._optim is not None: 
            self._optim.zero_grad()

        # -propagate through the network
        y_pred = self._model(x)
        # -calculate the loss

        loss = self._crit(y_pred, y) #TODO: not sure if this order is correct
        # -compute gradient by backward propagation
        loss.backward()

        # -update weights
        self._optim.step()

        # -return the loss
        return loss
        
    
    def val_test_step(self, x, y):
        
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        y_pred = self._model(x)
        loss = self._crit(y_pred, y) #TODO: not sure if this order is correct
        
        return loss, y_pred #TODO: Do the preds need to be transformed first?


    def train_epoch(self):
        # set training mode, #TODO: how exactly?
        # iterate through the training set
        loss_batches = []
        for batch in self._train_dl:
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            batch.to("cuda")
            # perform a training step. For this I need to know which shape batch exactly has

        # calculate the average loss for the epoch and return it
        
    
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        pass
        #TODO
        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
         # create a list for the train and validation losses, and create a counter for the epoch 
        #TODO
        
        while True:
      
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
            pass
        #TODO
                    
        
# The training process consists of alternating between training for one epoch on the training
# dataset (training step) and then assessing the performance on the validation dataset (validation
# step). After that, a decision is made if the training process should be continued. A common
# stopping criterion is called EarlyStopping with the following behaviour: If the validation loss
# does not decrease after a specified number of epochs, then the training process will be stopped.
# This criterion will be used in our implementation and should be realised in trainer.py.       