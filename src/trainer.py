import torch as t
from sklearn.metrics import f1_score, average_precision_score, recall_score
from tqdm.autonotebook import tqdm
import numpy as np


def _f1(prediction_batches):
    preds_binary = []
    labels = []

    for pred, label in prediction_batches:
        pred_bin = (pred >= 0.5).int() # we have probs but need binary data
        preds_binary.append(pred_bin)
        labels.append(label.int())

    #sklearn needs the data on cpu
    y_pred = t.cat(preds_binary, dim=0).cpu().numpy() 
    y_true = t.cat(labels, dim=0).cpu().numpy()

    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def _pr_auc(prediction_batches):
    preds_probs = []
    labels = []

    for pred, label in prediction_batches:
        preds_probs.append(pred)
        labels.append(label.int())

    y_prob = t.cat(preds_probs, dim=0).cpu().numpy()
    y_true = t.cat(labels, dim=0).cpu().numpy()

    return average_precision_score(y_true, y_prob, average="macro")


def _recall(prediction_batches):
    preds_binary = []
    labels = []

    for pred, label in prediction_batches:
        pred_bin = (pred >= 0.5).int()
        preds_binary.append(pred_bin)
        labels.append(label.int())

    y_pred = t.cat(preds_binary, dim=0).cpu().numpy()
    y_true = t.cat(labels, dim=0).cpu().numpy()

    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    recall_per_label = recall_score(y_true, y_pred, average=None, zero_division=0)
    return recall_macro, recall_per_label


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
        loss = self._crit(y_pred, y)
        # -compute gradient by backward propagation
        loss.backward()

        # -update weights
        if self._optim is not None: 
            self._optim.step()

        # -return the loss
        return loss
        
    
    def val_test_step(self, x, y):
        y_pred = self._model(x)
        loss = self._crit(y_pred, y)
        
        return loss, y_pred


    def train_epoch(self):
        # set training mode
        self._model.train() #this will activate training mode on all layers that support it
        # iterate through the training set
        loss_batches = []
        for x, y in self._train_dl:
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            x = x.cuda()
            y = y.cuda()
            # call train step function to get loss
            loss_batches.append(self.train_step(x, y).item())

        # calculate the average loss for the epoch and return it
        return np.mean(loss_batches)
        
 
    def val_test(self):
        self._model.eval()
        with t.no_grad(): #we do not need gradients
            prediction_batches = [] #stores (pred, label) per batch
            loss_batches = []
            # iterate through the validation set
            for x, y in self._val_test_dl:
                # transfer the batch to the gpu if given
                x = x.cuda()
                y = y.cuda()
                # perform a validation step
                loss, pred = self.val_test_step(x, y)
                loss_batches.append(loss.item())
                # save the predictions and the labels for each batch
                prediction_batches.append((pred, y))

        # F1 score, PR-AUC and Recall
        f1 = _f1(prediction_batches)
        pr_auc = _pr_auc(prediction_batches)
        recall_macro, recall_per_label = _recall(prediction_batches)

        print("F1: \n", f1)
        print("PR-AUC: \n", pr_auc)
        print("Recall Macro: \n", recall_macro)
        print("Recall per label [crack, inactive]: \n", recall_per_label)
        return np.mean(loss_batches)
        

    def _early_stop(self, curr_loss):
        if curr_loss < self.best_val_loss: 
            # we have an improvement so we do not stop
            self.best_val_loss = curr_loss
            self.no_improv_counter = 0 #reset counter to 0 bc we found better loss
            return False
            
        self.no_improv_counter += 1
        if self.no_improv_counter >= self._early_stopping_patience:
            # we need to stop
            return True

        # we still have some patience
        return False


    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        if epochs <= 0:
            return None, None

        train_losses = []
        val_losses = []
        counter = 0
        self.best_val_loss = float('inf') #for early stopping
        self.no_improv_counter = 0 #for early stopping
        
        while True:
            # stop by epoch number
            if counter >= epochs:
                break
            # train for a epoch 
            train_losses.append(self.train_epoch())
            # calculate the loss and metrics on the validation set
            val_loss = self.val_test()
            val_losses.append(val_loss)
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            self.save_checkpoint(counter)

            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if self._early_stopping_patience > 0 and self._early_stop(val_loss):
                break
            counter+=1
    
        return train_losses, val_losses