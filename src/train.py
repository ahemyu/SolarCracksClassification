import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
data = pd.read_csv("labels.csv", sep=";")
train_data, test_data = train_test_split(data, random_state=42) #TODO: Decide on train-test split
#split up training into training and validation
train_data, validation_data = train_test_split(train_data, test_size=0.1, random_state=42)

train_dataset = ChallengeDataset(train_data)
validation_dataset = ChallengeDataset(validation_data)
test_dataset = ChallengeDataset(test_data)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_dataloader = t.utils.data.DataLoader(
    train_dataset,
    batch_size=32,          # try 16 -> 32 -> 64 based on GPU memory
    shuffle=True,
    num_workers=8,          # then try 12
    pin_memory=True,        # useful when training on CUDA
    persistent_workers=True,
    prefetch_factor=2,
    drop_last=True
)

val_dataloader = t.utils.data.DataLoader(
    validation_dataset,
    batch_size=64,          # validation can usually be larger
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

# create an instance of our ResNet model
model = model.ResNet()
loss = t.nn.BCELoss() #TODO: try different ones
optimizer = t.optim.AdamW(params=model.parameters()) #TODO: choose lr
trainer = Trainer(model, loss, optimizer, train_dataloader, val_dataloader, cuda=True, early_stopping_patience=10) #TODO: research early stopping
# call fit on trainer
res = trainer.fit(100)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses/losses.png')